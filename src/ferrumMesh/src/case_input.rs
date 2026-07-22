use std::fmt::{self, Write as _};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt, OpenOptionsSyncExt};
use cap_std::ambient_authority;
use cap_std::fs::{Dir, OpenOptions};

use crate::{MeshError, Result};

const INPUT_LIMIT: u64 = 16 * 1024 * 1024;
const INPUT_PROBE_LIMIT: u64 = 16 * 1024 * 1024 + 1;

/// Capability-scoped reader for the small, trusted set of case inputs.
pub(crate) struct CaseInput {
    root: PathBuf,
}

enum OpenFailure {
    Root(io::Error),
    Ancestor(io::Error),
    FinalNotFound,
    FinalOther(io::Error),
}

enum OpenMode {
    Required,
    Optional,
}

impl CaseInput {
    pub(crate) fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
        }
    }

    pub(crate) fn required(&self, logical: &str) -> Result<String> {
        self.read_required(logical)
    }

    pub(crate) fn optional(&self, logical: &str) -> Result<Option<String>> {
        self.read_optional(logical)
    }

    fn read_required(&self, logical: &str) -> Result<String> {
        match self.read(logical, OpenMode::Required)? {
            Some(content) => Ok(content),
            None => Err(self.failure(logical, None)),
        }
    }

    fn read_optional(&self, logical: &str) -> Result<Option<String>> {
        self.read(logical, OpenMode::Optional)
    }

    fn read(&self, logical: &str, mode: OpenMode) -> Result<Option<String>> {
        self.read_with_hooks(logical, mode, || Ok(()), || Ok(()))
    }

    fn read_with_hooks<AfterPrecheck, AfterMetadata>(
        &self,
        logical: &str,
        mode: OpenMode,
        after_precheck: AfterPrecheck,
        after_initial_metadata: AfterMetadata,
    ) -> Result<Option<String>>
    where
        AfterPrecheck: FnOnce() -> io::Result<()>,
        AfterMetadata: FnOnce() -> io::Result<()>,
    {
        validate_logical(logical).map_err(|()| self.failure(logical, None))?;
        let mut pieces = logical.split('/').peekable();
        let root = Dir::open_ambient_dir(&self.root, ambient_authority())
            .map_err(OpenFailure::Root)
            .map_err(|failure| self.open_failure(logical, failure))?;
        let mut parent = root;
        let final_name = loop {
            let piece = pieces.next().ok_or_else(|| self.failure(logical, None))?;
            if pieces.peek().is_none() {
                break piece;
            }
            parent = parent
                .open_dir_nofollow(piece)
                .map_err(OpenFailure::Ancestor)
                .map_err(|failure| self.open_failure(logical, failure))?;
        };

        // Reject known special files before opening. The no-follow open below is
        // still authoritative and protects the final component from replacement.
        match parent.symlink_metadata(final_name) {
            Ok(metadata) if !metadata.is_file() => return Err(self.failure(logical, None)),
            Ok(_) => {}
            Err(error)
                if error.kind() == io::ErrorKind::NotFound
                    && matches!(mode, OpenMode::Optional) =>
            {
                return Ok(None);
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(self.open_failure(logical, OpenFailure::FinalNotFound));
            }
            Err(error) => return Err(self.open_failure(logical, OpenFailure::FinalOther(error))),
        }

        after_precheck().map_err(|error| self.failure(logical, Some(&error)))?;

        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        options.nonblock(true);
        let mut file = parent.open_with(final_name, &options).map_err(|error| {
            self.open_failure(
                logical,
                if error.kind() == io::ErrorKind::NotFound {
                    OpenFailure::FinalNotFound
                } else {
                    OpenFailure::FinalOther(error)
                },
            )
        })?;
        let before = file
            .metadata()
            .map_err(|error| self.failure(logical, Some(&error)))?;
        if !before.is_file() || before.len() > INPUT_LIMIT {
            return Err(self.failure(logical, None));
        }

        after_initial_metadata().map_err(|error| self.failure(logical, Some(&error)))?;

        let capacity = usize::try_from(before.len()).map_err(|_| MeshError::OutOfMemory)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|_| MeshError::OutOfMemory)?;
        read_bounded(&mut file, &mut bytes).map_err(|error| self.failure(logical, Some(&error)))?;
        let after = file
            .metadata()
            .map_err(|error| self.failure(logical, Some(&error)))?;
        if bytes.len() as u64 > INPUT_LIMIT
            || bytes.len() as u64 != before.len()
            || after.len() != before.len()
        {
            return Err(self.failure(logical, None));
        }
        String::from_utf8(bytes)
            .map(Some)
            .map_err(|_| self.failure(logical, None))
    }

    #[cfg(all(test, unix))]
    fn required_with_hooks<AfterPrecheck, AfterMetadata>(
        &self,
        logical: &str,
        after_precheck: AfterPrecheck,
        after_initial_metadata: AfterMetadata,
    ) -> Result<String>
    where
        AfterPrecheck: FnOnce() -> io::Result<()>,
        AfterMetadata: FnOnce() -> io::Result<()>,
    {
        match self.read_with_hooks(
            logical,
            OpenMode::Required,
            after_precheck,
            after_initial_metadata,
        )? {
            Some(content) => Ok(content),
            None => Err(self.failure(logical, None)),
        }
    }

    fn open_failure(&self, logical: &str, failure: OpenFailure) -> MeshError {
        match failure {
            OpenFailure::Root(error)
            | OpenFailure::Ancestor(error)
            | OpenFailure::FinalOther(error) => self.failure(logical, Some(&error)),
            OpenFailure::FinalNotFound => self.failure(logical, None),
        }
    }

    fn failure(&self, logical: &str, source: Option<&io::Error>) -> MeshError {
        let mut message = String::new();
        let source_len = match source {
            Some(error) => {
                let mut counter = CountingWriter::default();
                if write!(&mut counter, "{error}").is_err() {
                    return MeshError::OutOfMemory;
                }
                counter.len
            }
            None => 0,
        };
        let mut root_counter = CountingWriter::default();
        if write!(&mut root_counter, "{}", self.root.display()).is_err() {
            return MeshError::OutOfMemory;
        }
        let Some(needed) = root_counter
            .len
            .checked_add(logical.len())
            .and_then(|length| length.checked_add(source_len))
            .and_then(|length| length.checked_add(32))
        else {
            return MeshError::OutOfMemory;
        };
        if message.try_reserve(needed).is_err() {
            return MeshError::OutOfMemory;
        }
        if write!(
            &mut message,
            "could not read {}/{logical}",
            self.root.display()
        )
        .is_err()
        {
            return MeshError::OutOfMemory;
        }
        if let Some(error) = source
            && write!(&mut message, " ({error})").is_err()
        {
            return MeshError::OutOfMemory;
        }
        MeshError::InvalidInput(message)
    }
}

fn read_bounded(file: &mut cap_std::fs::File, bytes: &mut Vec<u8>) -> io::Result<()> {
    let mut chunk = [0_u8; 8192];
    loop {
        let read = file.read(&mut chunk)?;
        if read == 0 {
            return Ok(());
        }
        let next_len = bytes
            .len()
            .checked_add(read)
            .ok_or_else(|| io::Error::other("case input exceeds supported size"))?;
        if next_len as u64 > INPUT_PROBE_LIMIT {
            return Err(io::Error::other("case input exceeds 16 MiB limit"));
        }
        bytes
            .try_reserve_exact(read)
            .map_err(|_| io::Error::other("out of memory while reading case input"))?;
        bytes.extend_from_slice(&chunk[..read]);
    }
}

#[derive(Default)]
struct CountingWriter {
    len: usize,
}

impl fmt::Write for CountingWriter {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.len = self.len.checked_add(value.len()).ok_or(fmt::Error)?;
        Ok(())
    }
}

fn validate_logical(raw: &str) -> std::result::Result<(), ()> {
    if raw.is_empty()
        || raw.starts_with('/')
        || raw.ends_with('/')
        || raw.contains("//")
        || raw.contains('\\')
        || raw.contains(':')
    {
        return Err(());
    }
    for piece in raw.split('/') {
        if piece.is_empty() || piece == "." || piece == ".." {
            return Err(());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::read_backend_config;
    use crate::control::read_control_dict;
    use crate::interfaces::read_interface_config;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempCase {
        root: PathBuf,
    }

    impl TempCase {
        fn new(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "ferrum-case-input-{name}-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&root).unwrap();
            Self { root }
        }

        fn mkdir(&self, logical: &str) {
            fs::create_dir_all(self.root.join(logical)).unwrap();
        }

        fn write(&self, logical: &str, bytes: &[u8]) {
            let path = self.root.join(logical);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(path, bytes).unwrap();
        }
    }

    impl Drop for TempCase {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn display_error(error: MeshError) -> String {
        error.to_string()
    }

    #[test]
    fn raw_paths_reject_windows_prefixes() {
        let case = TempCase::new("raw-paths");
        let input = CaseInput::new(&case.root);

        for logical in [
            "",
            "/system/controlDict",
            "system/",
            "system//controlDict",
            "system\\controlDict",
            "C:/system/controlDict",
            "../system/controlDict",
            "system/../controlDict",
            "system/./controlDict",
        ] {
            let error = display_error(input.required(logical).unwrap_err());
            assert!(error.contains(&case.root.display().to_string()));
            assert!(error.contains(logical));
        }
    }

    #[test]
    fn exact_input_cap_and_one_over() {
        let case = TempCase::new("cap");
        case.mkdir("system");
        let exact = vec![b'a'; usize::try_from(INPUT_LIMIT).unwrap()];
        case.write("system/exact", &exact);
        assert_eq!(
            CaseInput::new(&case.root)
                .required("system/exact")
                .unwrap()
                .len(),
            exact.len()
        );

        let too_large = case.root.join("system/too-large");
        let file = fs::File::create(&too_large).unwrap();
        file.set_len(INPUT_LIMIT + 1).unwrap();
        let error = display_error(
            CaseInput::new(&case.root)
                .required("system/too-large")
                .unwrap_err(),
        );
        assert!(error.contains("system/too-large"));
    }

    #[test]
    fn opened_handle_growth_and_shrink_fail() {
        let case = TempCase::new("metadata");
        case.write("system/controlDict", b"application ferrum;\n");
        let input = CaseInput::new(&case.root);
        assert_eq!(
            input.required("system/controlDict").unwrap(),
            "application ferrum;\n"
        );

        #[cfg(unix)]
        {
            let shrink = case.root.join("system/shrink");
            fs::write(&shrink, b"application ferrum;\n").unwrap();
            let error = display_error(
                input
                    .required_with_hooks(
                        "system/shrink",
                        || Ok(()),
                        || {
                            fs::write(&shrink, b"")?;
                            Ok(())
                        },
                    )
                    .unwrap_err(),
            );
            assert!(error.contains("system/shrink"));
        }

        let oversized = case.root.join("system/oversized");
        let file = fs::File::create(&oversized).unwrap();
        file.set_len(INPUT_LIMIT + 1).unwrap();
        let error = display_error(input.required("system/oversized").unwrap_err());
        assert!(error.contains("system/oversized"));
    }

    #[test]
    fn replacement_uses_original_handle() {
        let case = TempCase::new("replacement");
        case.write("system/controlDict", b"application ferrum;\n");
        let first = CaseInput::new(&case.root)
            .required("system/controlDict")
            .unwrap();
        fs::write(
            case.root.join("system/controlDict"),
            b"application replaced;\n",
        )
        .unwrap();
        let second = CaseInput::new(&case.root)
            .required("system/controlDict")
            .unwrap();
        assert_eq!(first, "application ferrum;\n");
        assert_eq!(second, "application replaced;\n");

        #[cfg(unix)]
        {
            let path = case.root.join("system/raced");
            let replacement = case.root.join("system/raced.new");
            fs::write(&path, b"application original;\n").unwrap();
            let raced = CaseInput::new(&case.root)
                .required_with_hooks(
                    "system/raced",
                    || Ok(()),
                    || {
                        fs::write(&replacement, b"application replacement;\n")?;
                        fs::rename(&replacement, &path)
                    },
                )
                .unwrap();
            assert_eq!(raced, "application original;\n");
        }
    }

    #[test]
    fn invalid_utf8_is_sticky_without_prefix() {
        let case = TempCase::new("utf8");
        case.write("system/controlDict", b"application ferrum;\n\xff");
        let input = CaseInput::new(&case.root);
        let first = display_error(input.required("system/controlDict").unwrap_err());
        let second = display_error(input.required("system/controlDict").unwrap_err());
        assert_eq!(first, second);
        assert!(first.contains("system/controlDict"));
        assert!(!first.contains("application ferrum"));
    }

    #[cfg(unix)]
    #[test]
    fn unix_symlink_escapes_fail_closed() {
        use std::os::unix::fs::symlink;
        let case = TempCase::new("symlink");
        let outside = TempCase::new("outside");
        outside.write("controlDict", b"application outside;\n");
        case.mkdir("system");
        symlink(
            outside.root.join("controlDict"),
            case.root.join("system/controlDict"),
        )
        .unwrap();
        let error = display_error(
            CaseInput::new(&case.root)
                .required("system/controlDict")
                .unwrap_err(),
        );
        assert!(error.contains("system/controlDict"));
        assert!(!error.contains("outside"));
    }

    #[cfg(unix)]
    #[test]
    fn unix_socket_is_rejected() {
        use std::os::unix::net::UnixListener;
        let case = TempCase::new("socket");
        case.mkdir("system");
        let socket = case.root.join("system/controlDict");
        let _listener = UnixListener::bind(&socket).unwrap();
        let error = display_error(
            CaseInput::new(&case.root)
                .required("system/controlDict")
                .unwrap_err(),
        );
        assert!(error.contains("system/controlDict"));
    }

    #[cfg(unix)]
    #[test]
    fn unix_fifo_is_bounded() {
        use std::process::Command;
        use std::time::{Duration, Instant};
        let case = TempCase::new("fifo");
        case.mkdir("system");
        let fifo = case.root.join("system/controlDict");
        let status = Command::new("mkfifo").arg(&fifo).status().unwrap();
        assert!(status.success());
        let start = Instant::now();
        let error = display_error(
            CaseInput::new(&case.root)
                .required("system/controlDict")
                .unwrap_err(),
        );
        assert!(start.elapsed() < Duration::from_secs(2));
        assert!(error.contains("system/controlDict"));
    }

    #[cfg(windows)]
    #[test]
    fn windows_junction_escapes_fail_closed() {
        use std::process::Command;
        let case = TempCase::new("junction");
        let outside = TempCase::new("outside");
        outside.write("controlDict", b"application outside;\n");
        case.mkdir("system");
        let junction = case.root.join("system/controlDict");
        let output = Command::new("cmd")
            .args([
                "/C",
                "mklink",
                "/J",
                junction.to_str().unwrap(),
                outside.root.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        if !output.status.success() {
            return;
        }
        let error = display_error(
            CaseInput::new(&case.root)
                .required("system/controlDict")
                .unwrap_err(),
        );
        assert!(error.contains("system/controlDict"));
        assert!(!error.contains("outside"));
    }

    #[test]
    fn control_guidance_is_exact_once() {
        let case = TempCase::new("control-guidance");
        let error = display_error(read_control_dict(&case.root).unwrap_err());
        assert_eq!(error.matches("run initFerrumCase first").count(), 1);
        assert!(error.contains("system/controlDict"));
    }

    #[test]
    fn backend_caller_reports_case_input_scope() {
        let case = TempCase::new("backend-caller");
        fs::write(case.root.join("system"), b"not a directory").unwrap();
        let error = display_error(read_backend_config(&case.root).unwrap_err());
        assert!(error.contains("system/ferrumBackends"));
        assert!(error.contains(&case.root.display().to_string()));
    }

    #[test]
    fn interface_caller_reports_case_input_scope() {
        let case = TempCase::new("interface-caller");
        fs::write(case.root.join("constant"), b"not a directory").unwrap();
        let error = display_error(read_interface_config(&case.root).unwrap_err());
        assert!(error.contains("constant/interfaces"));
        assert!(error.contains(&case.root.display().to_string()));
    }

    #[test]
    fn optional_callers_distinguish_final_and_ancestor() {
        let case = TempCase::new("optional");
        case.mkdir("system");
        assert!(read_backend_config(&case.root).unwrap().is_none());

        let blocked = TempCase::new("blocked-ancestor");
        fs::write(blocked.root.join("system"), b"not a directory").unwrap();
        let error = display_error(read_backend_config(&blocked.root).unwrap_err());
        assert!(error.contains("system/ferrumBackends"));
    }
}
