use std::ffi::{OsStr, OsString};
use std::io;
use std::path::{Component, Path, PathBuf};

use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::ambient_authority;
use cap_std::fs::{Dir, OpenOptions};

/// The no-follow kind of an entry below a [`SafeOutputRoot`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SafeOutputEntry {
    File,
    Directory,
    Other,
}

/// A capability-scoped output directory whose descendants are opened one
/// component at a time without following links or Windows reparse points.
pub struct SafeOutputRoot {
    dir: Dir,
    path: PathBuf,
}

impl SafeOutputRoot {
    /// Opens an existing directory without following any ancestor links.
    pub fn open_existing(path: &Path) -> io::Result<Self> {
        let (dir, absolute, _) = open_absolute_directory_strict(path, false)?;
        Ok(Self {
            dir,
            path: absolute,
        })
    }

    /// Opens or creates a directory without following any ancestor links.
    pub fn create(path: &Path) -> io::Result<(Self, Vec<PathBuf>)> {
        let (dir, absolute, created) = open_absolute_directory_strict(path, true)?;
        Ok((
            Self {
                dir,
                path: absolute,
            },
            created,
        ))
    }

    /// Opens an existing operator-supplied directory as a trusted root.
    ///
    /// Links in `path` itself are followed while acquiring the root
    /// capability. Every relative descent below that capability remains
    /// no-follow.
    pub fn open_trusted(path: &Path) -> io::Result<Self> {
        let (dir, absolute, _) = open_absolute_directory_trusted(path, false)?;
        Ok(Self {
            dir,
            path: absolute,
        })
    }

    /// Opens or creates an operator-supplied directory as a trusted root.
    ///
    /// An existing prefix of `path` may contain links. Missing components
    /// below the opened prefix are created one at a time with no-follow
    /// checks. The returned paths are the directories created by this call.
    pub fn create_trusted(path: &Path) -> io::Result<(Self, Vec<PathBuf>)> {
        let (dir, absolute, created) = open_absolute_directory_trusted(path, true)?;
        Ok((
            Self {
                dir,
                path: absolute,
            },
            created,
        ))
    }

    /// Creates a new file at an operator-supplied absolute path.
    ///
    /// Missing parent directories are created below the deepest trusted
    /// existing ancestor. If file creation fails, directories created by this
    /// call are removed through their retained parent capabilities.
    pub fn open_create_new_trusted_absolute(path: &Path) -> io::Result<std::fs::File> {
        if !path.is_absolute() {
            return Err(invalid_path(
                "trusted operator output path must be absolute",
            ));
        }
        let absolute = lexical_absolute(path)?;
        validate_absolute(&absolute)?;
        let file_name = absolute
            .file_name()
            .ok_or_else(|| invalid_path("absolute output path has no file name"))?;
        validate_name(file_name)?;
        let parent = absolute
            .parent()
            .ok_or_else(|| invalid_path("absolute output path has no parent directory"))?;
        let (dir, parent_path, created) =
            open_absolute_directory_trusted_with_handles(parent, true)?;
        let root = Self {
            dir,
            path: parent_path,
        };
        match root.open_create_new(Path::new(file_name)) {
            Ok(file) => Ok(file),
            Err(error) => {
                drop(root);
                rollback_created_directories(created);
                Err(error)
            }
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Validates a relative file path without touching the filesystem.
    pub fn validate_file_path(&self, relative: &Path) -> io::Result<()> {
        validate_relative(relative, false)
    }

    /// Ensures a relative directory exists below this root and returns the
    /// absolute paths of directories created by this call.
    pub fn ensure_dir(&self, relative: &Path) -> io::Result<Vec<PathBuf>> {
        validate_relative(relative, true)?;
        let mut current = self.dir.try_clone()?;
        let mut display = try_path_buf(&self.path)?;
        let mut created = Vec::new();

        for component in relative.components() {
            let Component::Normal(piece) = component else {
                continue;
            };
            try_push_path(&mut display, Path::new(piece))?;
            current = open_or_create_child(&current, piece, true, &display, &mut created)?;
        }
        Ok(created)
    }

    /// Returns the no-follow kind of a relative entry, or `None` if absent.
    pub fn entry(&self, relative: &Path) -> io::Result<Option<SafeOutputEntry>> {
        let (parent, name) = self.open_parent(relative, false)?;
        match parent.symlink_metadata(name) {
            Ok(metadata) if metadata.is_file() => Ok(Some(SafeOutputEntry::File)),
            Ok(metadata) if metadata.is_dir() => Ok(Some(SafeOutputEntry::Directory)),
            Ok(_) => Ok(Some(SafeOutputEntry::Other)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Creates a new regular file, creating safe parent directories as needed.
    /// Existing entries are never overwritten.
    pub fn open_create_new(&self, relative: &Path) -> io::Result<std::fs::File> {
        let (parent, name) = self.open_parent(relative, true)?;
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let file = parent.open_with(name, &options)?;
        ensure_regular_file(&file)?;
        Ok(file.into_std())
    }

    /// Replaces a regular file without following or modifying another inode.
    ///
    /// An existing entry is first classified without following it and then
    /// unlinked.  The replacement itself is always created with `create_new`.
    /// This avoids blocking on special files and prevents a hard link from
    /// truncating data outside this capability root.
    pub fn open_replace_regular(&self, relative: &Path) -> io::Result<std::fs::File> {
        let (parent, name) = self.open_parent(relative, false)?;
        match parent.symlink_metadata(name) {
            Ok(metadata) if metadata.is_file() => parent.remove_file(name)?,
            Ok(_) => return Err(invalid_path("output path is not a regular file")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }

        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let file = parent.open_with(name, &options)?;
        ensure_regular_file(&file)?;
        Ok(file.into_std())
    }

    fn open_parent<'a>(&self, relative: &'a Path, create: bool) -> io::Result<(Dir, &'a OsStr)> {
        validate_relative(relative, false)?;
        let name = relative
            .file_name()
            .ok_or_else(|| invalid_path("output path has no file name"))?;
        validate_name(name)?;
        let parent_path = relative.parent().unwrap_or_else(|| Path::new(""));
        let mut parent = self.dir.try_clone()?;
        let mut display = try_path_buf(&self.path)?;
        let mut ignored_created = Vec::new();

        for component in parent_path.components() {
            let Component::Normal(piece) = component else {
                continue;
            };
            try_push_path(&mut display, Path::new(piece))?;
            parent = open_or_create_child(&parent, piece, create, &display, &mut ignored_created)?;
        }
        Ok((parent, name))
    }
}

fn open_absolute_directory_strict(
    path: &Path,
    create: bool,
) -> io::Result<(Dir, PathBuf, Vec<PathBuf>)> {
    let absolute = lexical_absolute(path)?;
    validate_absolute(&absolute)?;

    let anchor = absolute_anchor(&absolute)?;
    let mut current = Dir::open_ambient_dir(&anchor, ambient_authority())?;
    let mut display = anchor;
    let mut created = Vec::new();

    for component in absolute.components() {
        let Component::Normal(piece) = component else {
            continue;
        };
        try_push_path(&mut display, Path::new(piece))?;
        current = open_or_create_child(&current, piece, create, &display, &mut created)?;
    }
    Ok((current, display, created))
}

fn open_absolute_directory_trusted(
    path: &Path,
    create: bool,
) -> io::Result<(Dir, PathBuf, Vec<PathBuf>)> {
    let (dir, absolute, created) = open_absolute_directory_trusted_with_handles(path, create)?;
    let mut created_paths = Vec::new();
    if created_paths.try_reserve_exact(created.len()).is_err() {
        drop(dir);
        rollback_created_directories(created);
        return Err(out_of_memory());
    }
    for entry in created {
        created_paths.push(entry.path);
    }
    Ok((dir, absolute, created_paths))
}

fn open_absolute_directory_trusted_with_handles(
    path: &Path,
    create: bool,
) -> io::Result<(Dir, PathBuf, Vec<CreatedRootDirectory>)> {
    let absolute = lexical_absolute(path)?;
    validate_absolute(&absolute)?;

    match Dir::open_ambient_dir(&absolute, ambient_authority()) {
        Ok(dir) => return Ok((dir, absolute, Vec::new())),
        Err(error) if error.kind() == io::ErrorKind::NotFound && create => {}
        Err(error) => return Err(error),
    }

    let (mut current, mut display, missing) = open_deepest_trusted_ancestor(&absolute)?;
    let mut created = Vec::new();
    created
        .try_reserve_exact(missing.len())
        .map_err(|_| out_of_memory())?;
    for piece in missing.iter().rev() {
        if let Err(error) = try_push_path(&mut display, Path::new(piece)) {
            drop(current);
            rollback_created_directories(created);
            return Err(error);
        }
        match open_or_create_root_child(&current, piece, &display, &mut created) {
            Ok(child) => current = child,
            Err(error) => {
                drop(current);
                rollback_created_directories(created);
                return Err(error);
            }
        }
    }
    Ok((current, absolute, created))
}

/// Opens the deepest existing prefix of an operator-supplied trusted root.
/// The returned missing components are ordered from the requested leaf back
/// toward that prefix so callers can create them by iterating in reverse.
fn open_deepest_trusted_ancestor(path: &Path) -> io::Result<(Dir, PathBuf, Vec<OsString>)> {
    let mut candidate = try_path_buf(path)?;
    let mut missing = Vec::new();

    loop {
        match Dir::open_ambient_dir(&candidate, ambient_authority()) {
            Ok(dir) => return Ok((dir, candidate, missing)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }

        let name = try_os_string(
            candidate
                .file_name()
                .ok_or_else(|| invalid_path("could not find an existing output root ancestor"))?,
        )?;
        if !candidate.pop() {
            return Err(invalid_path(
                "could not find an existing output root ancestor",
            ));
        }
        missing.try_reserve(1).map_err(|_| out_of_memory())?;
        missing.push(name);
    }
}

struct CreatedRootDirectory {
    parent: Dir,
    name: OsString,
    path: PathBuf,
}

fn open_or_create_root_child(
    parent: &Dir,
    name: &OsStr,
    display: &Path,
    created: &mut Vec<CreatedRootDirectory>,
) -> io::Result<Dir> {
    match parent.open_dir_nofollow(name) {
        Ok(dir) => Ok(dir),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            created.try_reserve(1).map_err(|_| out_of_memory())?;
            let rollback_parent = parent.try_clone()?;
            let rollback_name = try_os_string(name)?;
            let rollback_path = try_path_buf(display)?;
            match parent.create_dir(name) {
                Ok(()) => {
                    let child = match parent.open_dir_nofollow(name) {
                        Ok(child) => child,
                        Err(error) => {
                            let _ = rollback_parent.remove_dir(name);
                            return Err(error);
                        }
                    };
                    created.push(CreatedRootDirectory {
                        parent: rollback_parent,
                        name: rollback_name,
                        path: rollback_path,
                    });
                    Ok(child)
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    // The authoritative no-follow open rejects a link or
                    // reparse point installed by a racing process.
                    parent.open_dir_nofollow(name)
                }
                Err(error) => Err(error),
            }
        }
        Err(error) => Err(error),
    }
}

fn rollback_created_directories(mut created: Vec<CreatedRootDirectory>) {
    while let Some(entry) = created.pop() {
        let _ = entry.parent.remove_dir(&entry.name);
    }
}

fn lexical_absolute(path: &Path) -> io::Result<PathBuf> {
    let joined = if path.is_absolute() {
        try_path_buf(path)?
    } else {
        let mut current = std::env::current_dir()?;
        try_push_path(&mut current, path)?;
        current
    };
    let mut normalized = PathBuf::new();
    normalized
        .try_reserve(joined.as_os_str().len())
        .map_err(|_| out_of_memory())?;
    for component in joined.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::Normal(piece) => normalized.push(piece),
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(invalid_path("output path escapes its filesystem root"));
                }
            }
        }
    }
    if normalized.is_absolute() {
        Ok(normalized)
    } else {
        Err(invalid_path("output root must resolve to an absolute path"))
    }
}

fn open_or_create_child(
    parent: &Dir,
    name: &OsStr,
    create: bool,
    display: &Path,
    created: &mut Vec<PathBuf>,
) -> io::Result<Dir> {
    match parent.open_dir_nofollow(name) {
        Ok(dir) => Ok(dir),
        Err(error) if error.kind() == io::ErrorKind::NotFound && create => {
            created.try_reserve(1).map_err(|_| out_of_memory())?;
            let created_path = try_path_buf(display)?;
            match parent.create_dir(name) {
                Ok(()) => created.push(created_path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
            // This second no-follow open is authoritative if another process
            // raced the creation with a link or reparse point.
            parent.open_dir_nofollow(name)
        }
        Err(error) => Err(error),
    }
}

fn ensure_regular_file(file: &cap_std::fs::File) -> io::Result<()> {
    if file.metadata()?.is_file() {
        Ok(())
    } else {
        Err(invalid_path("output path is not a regular file"))
    }
}

fn validate_absolute(path: &Path) -> io::Result<()> {
    if !path.is_absolute() {
        return Err(invalid_path("output root must resolve to an absolute path"));
    }
    for component in path.components() {
        match component {
            Component::ParentDir => {
                return Err(invalid_path("parent-directory components are not allowed"));
            }
            Component::Normal(name) => validate_name(name)?,
            Component::Prefix(_) | Component::RootDir | Component::CurDir => {}
        }
    }
    Ok(())
}

fn validate_relative(path: &Path, allow_empty: bool) -> io::Result<()> {
    if path.is_absolute() {
        return Err(invalid_path(
            "absolute output paths are not allowed below a root",
        ));
    }
    let mut normal_count = 0usize;
    for component in path.components() {
        match component {
            Component::Normal(name) => {
                validate_name(name)?;
                normal_count = normal_count
                    .checked_add(1)
                    .ok_or_else(|| invalid_path("too many output path components"))?;
            }
            Component::CurDir => {}
            Component::ParentDir | Component::Prefix(_) | Component::RootDir => {
                return Err(invalid_path("output path escapes its capability root"));
            }
        }
    }
    if normal_count == 0 && !allow_empty {
        return Err(invalid_path("output path must name a file"));
    }
    Ok(())
}

fn validate_name(name: &OsStr) -> io::Result<()> {
    if name.is_empty() {
        return Err(invalid_path("empty output path component"));
    }
    #[cfg(windows)]
    if name.to_string_lossy().contains(':') {
        return Err(invalid_path(
            "Windows alternate data streams are not allowed",
        ));
    }
    Ok(())
}

fn absolute_anchor(path: &Path) -> io::Result<PathBuf> {
    let mut anchor = PathBuf::new();
    anchor
        .try_reserve(path.as_os_str().len())
        .map_err(|_| out_of_memory())?;
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => anchor.push(prefix.as_os_str()),
            Component::RootDir => anchor.push(component.as_os_str()),
            Component::CurDir => {}
            Component::Normal(_) => break,
            Component::ParentDir => {
                return Err(invalid_path("parent-directory components are not allowed"));
            }
        }
    }
    if anchor.as_os_str().is_empty() {
        Err(invalid_path("could not determine output path anchor"))
    } else {
        Ok(anchor)
    }
}

fn try_os_string(value: &OsStr) -> io::Result<OsString> {
    let mut owned = OsString::new();
    owned
        .try_reserve(value.len())
        .map_err(|_| out_of_memory())?;
    owned.push(value);
    Ok(owned)
}

fn try_path_buf(value: &Path) -> io::Result<PathBuf> {
    let mut owned = PathBuf::new();
    owned
        .try_reserve(value.as_os_str().len())
        .map_err(|_| out_of_memory())?;
    owned.push(value);
    Ok(owned)
}

fn try_push_path(target: &mut PathBuf, value: &Path) -> io::Result<()> {
    let reserve = value
        .as_os_str()
        .len()
        .checked_add(1)
        .ok_or_else(out_of_memory)?;
    target.try_reserve(reserve).map_err(|_| out_of_memory())?;
    target.push(value);
    Ok(())
}

fn out_of_memory() -> io::Error {
    io::Error::from(io::ErrorKind::OutOfMemory)
}

fn invalid_path(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn create_new_does_not_clobber_existing_file() {
        let base = temp_dir("create-new");
        let (root, _) = SafeOutputRoot::create(&base).expect("create root");
        let mut first = root
            .open_create_new(Path::new("nested/result.json"))
            .expect("create output");
        first.write_all(b"original").expect("write output");
        drop(first);

        let error = root
            .open_create_new(Path::new("nested/result.json"))
            .expect_err("existing output must not be clobbered");
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(
            std::fs::read(base.join("nested/result.json")).expect("read output"),
            b"original"
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn trusted_absolute_create_new_rolls_back_new_parents_on_leaf_failure() {
        let base = temp_dir("trusted-absolute-rollback");
        std::fs::create_dir_all(&base).expect("create base");
        let created_parent = base.join("new-parent");
        let oversized_leaf = "x".repeat(300);
        let path = created_parent.join(oversized_leaf);

        SafeOutputRoot::open_create_new_trusted_absolute(&path)
            .expect_err("oversized leaf must fail after safe parent creation");

        assert!(!created_parent.exists());
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn rollback_preserves_a_created_directory_that_became_nonempty() {
        let base = temp_dir("rollback-nonempty");
        std::fs::create_dir_all(&base).expect("create base");
        let parent = Dir::open_ambient_dir(&base, ambient_authority()).expect("open base");
        parent.create_dir("created").expect("create directory");
        let created_path = base.join("created");
        std::fs::write(created_path.join("sentinel"), b"preserve").expect("write sentinel");
        rollback_created_directories(vec![CreatedRootDirectory {
            parent,
            name: OsString::from("created"),
            path: created_path.clone(),
        }]);
        assert_eq!(
            std::fs::read(created_path.join("sentinel")).unwrap(),
            b"preserve"
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn replace_creates_a_new_regular_file() {
        let base = temp_dir("replace");
        let (root, _) = SafeOutputRoot::create(&base).expect("create root");
        std::fs::write(base.join("value"), "old").expect("write fixture");
        let mut file = root
            .open_replace_regular(Path::new("value"))
            .expect("replace output");
        file.write_all(b"new").expect("write replacement");
        drop(file);

        let mut contents = String::new();
        std::fs::File::open(base.join("value"))
            .expect("open value")
            .read_to_string(&mut contents)
            .expect("read value");
        assert_eq!(contents, "new");
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn replace_does_not_modify_an_external_hard_link_target() {
        let base = temp_dir("replace-hard-link");
        let outside = temp_dir("replace-hard-link-outside");
        std::fs::create_dir_all(&base).expect("create base");
        std::fs::create_dir_all(&outside).expect("create outside");
        let target = outside.join("target");
        std::fs::write(&target, "external").expect("write target");
        std::fs::hard_link(&target, base.join("value")).expect("create hard link");

        let root = SafeOutputRoot::open_existing(&base).expect("open root");
        let mut file = root
            .open_replace_regular(Path::new("value"))
            .expect("replace hard link entry");
        file.write_all(b"local").expect("write replacement");
        drop(file);

        assert_eq!(std::fs::read_to_string(&target).unwrap(), "external");
        assert_eq!(
            std::fs::read_to_string(base.join("value")).unwrap(),
            "local"
        );
        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[cfg(unix)]
    #[test]
    fn replace_rejects_fifo_without_opening_it() {
        let base = temp_dir("replace-fifo");
        std::fs::create_dir_all(&base).expect("create base");
        let fifo = base.join("value");
        let status = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("run mkfifo");
        assert!(status.success(), "mkfifo must create the test fixture");

        let root = SafeOutputRoot::open_existing(&base).expect("open root");
        assert!(root.open_replace_regular(Path::new("value")).is_err());
        assert!(fifo.exists());
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn rejects_parent_directory_escape() {
        let base = temp_dir("parent");
        let (root, _) = SafeOutputRoot::create(&base).expect("create root");
        let error = root
            .open_create_new(Path::new("../outside"))
            .expect_err("parent escape must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        let _ = std::fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_output_ancestor() {
        use std::os::unix::fs::symlink;

        let base = temp_dir("symlink-ancestor");
        let outside = temp_dir("symlink-outside");
        std::fs::create_dir_all(&base).expect("create base");
        std::fs::create_dir_all(&outside).expect("create outside");
        symlink(&outside, base.join("linked")).expect("create symlink");
        let root = SafeOutputRoot::open_existing(&base).expect("open root");
        assert!(root.open_create_new(Path::new("linked/result")).is_err());
        assert!(!outside.join("result").exists());
        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[cfg(unix)]
    #[test]
    fn opens_trusted_symlink_root_but_rejects_links_below_it() {
        use std::os::unix::fs::symlink;

        let base = temp_dir("trusted-symlink-root");
        let target = temp_dir("trusted-symlink-target");
        let outside = temp_dir("trusted-symlink-outside");
        std::fs::create_dir_all(&base).expect("create base");
        std::fs::create_dir_all(&target).expect("create target root");
        std::fs::create_dir_all(&outside).expect("create outside");
        symlink(&target, base.join("trusted")).expect("create trusted ancestor link");

        let trusted_root = base.join("trusted");
        assert!(SafeOutputRoot::open_existing(&trusted_root).is_err());
        let root = SafeOutputRoot::open_trusted(&trusted_root).expect("open trusted linked root");
        root.open_create_new(Path::new("result"))
            .expect("create inside trusted root");
        assert_eq!(root.path(), trusted_root);
        assert!(target.join("result").is_file());

        symlink(&outside, target.join("linked")).expect("create descendant link");
        assert!(root.open_create_new(Path::new("linked/escaped")).is_err());
        assert!(!outside.join("escaped").exists());

        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(target);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[cfg(unix)]
    #[test]
    fn creates_missing_root_below_trusted_symlinked_ancestor() {
        use std::os::unix::fs::symlink;

        let base = temp_dir("create-trusted-symlink-ancestor");
        let target = temp_dir("create-trusted-symlink-target");
        std::fs::create_dir_all(&base).expect("create base");
        std::fs::create_dir_all(&target).expect("create target");
        symlink(&target, base.join("trusted")).expect("create trusted ancestor link");

        let trusted_root = base.join("trusted/new/nested");
        let (root, created) =
            SafeOutputRoot::create_trusted(&trusted_root).expect("create linked root");
        root.open_create_new(Path::new("result"))
            .expect("create inside linked root");

        assert_eq!(
            created,
            vec![base.join("trusted/new"), base.join("trusted/new/nested")]
        );
        assert!(target.join("new/nested/result").is_file());

        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(target);
    }

    #[cfg(windows)]
    #[test]
    fn rejects_windows_reparse_output_ancestor() {
        use std::os::windows::fs::symlink_dir;

        let base = temp_dir("reparse-ancestor");
        let outside = temp_dir("reparse-outside");
        std::fs::create_dir_all(&base).expect("create base");
        std::fs::create_dir_all(&outside).expect("create outside");
        if symlink_dir(&outside, base.join("linked")).is_err() {
            let _ = std::fs::remove_dir_all(base);
            let _ = std::fs::remove_dir_all(outside);
            return;
        }
        let root = SafeOutputRoot::open_existing(&base).expect("open root");
        assert!(root.open_create_new(Path::new("linked/result")).is_err());
        assert!(!outside.join("result").exists());
        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[cfg(windows)]
    #[test]
    fn rejects_windows_junction_output_ancestor() {
        let base = temp_dir("junction-ancestor");
        let outside = temp_dir("junction-outside");
        std::fs::create_dir_all(&base).expect("create base");
        std::fs::create_dir_all(&outside).expect("create outside");
        let junction = base.join("linked");
        if create_windows_junction(&outside, &junction).is_err() {
            let _ = std::fs::remove_dir_all(base);
            let _ = std::fs::remove_dir_all(outside);
            return;
        }

        let root = SafeOutputRoot::open_existing(&base).expect("open root");
        assert!(root.open_create_new(Path::new("linked/result")).is_err());
        assert!(!outside.join("result").exists());

        let _ = std::fs::remove_dir(&junction);
        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[cfg(windows)]
    #[test]
    fn accepts_windows_junction_as_trusted_output_root() {
        let base = temp_dir("junction-root");
        let outside = temp_dir("junction-root-outside");
        std::fs::create_dir_all(&base).expect("create base");
        std::fs::create_dir_all(&outside).expect("create outside");
        let junction = base.join("root");
        if create_windows_junction(&outside, &junction).is_err() {
            let _ = std::fs::remove_dir_all(base);
            let _ = std::fs::remove_dir_all(outside);
            return;
        }

        assert!(SafeOutputRoot::open_existing(&junction).is_err());
        let root = SafeOutputRoot::open_trusted(&junction).expect("open trusted junction root");
        root.open_create_new(Path::new("result"))
            .expect("create inside trusted junction root");
        assert!(outside.join("result").is_file());

        let _ = std::fs::remove_dir(&junction);
        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[cfg(windows)]
    #[test]
    fn creates_missing_root_below_trusted_windows_junction() {
        let base = temp_dir("create-junction-root");
        let target = temp_dir("create-junction-target");
        std::fs::create_dir_all(&base).expect("create base");
        std::fs::create_dir_all(&target).expect("create target");
        let junction = base.join("trusted");
        if create_windows_junction(&target, &junction).is_err() {
            let _ = std::fs::remove_dir_all(base);
            let _ = std::fs::remove_dir_all(target);
            return;
        }

        let trusted_root = junction.join("new/nested");
        let (root, created) =
            SafeOutputRoot::create_trusted(&trusted_root).expect("create linked root");
        root.open_create_new(Path::new("result"))
            .expect("create inside linked root");
        assert_eq!(
            created,
            vec![junction.join("new"), junction.join("new/nested")]
        );
        assert!(target.join("new/nested/result").is_file());

        let _ = std::fs::remove_dir(&junction);
        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(target);
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn replace_rejects_final_symlink_without_clobbering_target() {
        let base = temp_dir("final-symlink");
        std::fs::create_dir_all(&base).expect("create base");
        let target = base.join("target");
        let link = base.join("link");
        std::fs::write(&target, "original").expect("write target");
        if create_file_symlink(&target, &link).is_err() {
            let _ = std::fs::remove_dir_all(base);
            return;
        }

        let root = SafeOutputRoot::open_existing(&base).expect("open root");
        assert!(root.open_replace_regular(Path::new("link")).is_err());
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "original");
        let _ = std::fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    fn create_file_symlink(target: &Path, link: &Path) -> io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    #[cfg(windows)]
    fn create_file_symlink(target: &Path, link: &Path) -> io::Result<()> {
        std::os::windows::fs::symlink_file(target, link)
    }

    #[cfg(windows)]
    fn create_windows_junction(target: &Path, link: &Path) -> io::Result<()> {
        let status = std::process::Command::new("cmd")
            .arg("/C")
            .arg("mklink")
            .arg("/J")
            .arg(link)
            .arg(target)
            .status()?;
        if status.success() {
            Ok(())
        } else {
            Err(io::Error::other("mklink /J failed"))
        }
    }

    fn temp_dir(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("ferrum-safe-output-{label}-{unique}"))
    }
}
