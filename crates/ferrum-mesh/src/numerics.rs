use std::fs;
use std::path::{Path, PathBuf};

use crate::dictionary::{TokenCursor, tokenize};
use crate::{MeshError, Result};

#[derive(Debug)]
pub struct FvSchemes {
    pub path: PathBuf,
    pub sections: Vec<NumericsSection>,
}

#[derive(Debug)]
pub struct FvSolution {
    pub path: PathBuf,
    pub sections: Vec<NumericsSection>,
}

#[derive(Clone, Debug)]
pub struct NumericsSection {
    pub name: String,
    pub entries: Vec<NumericsEntry>,
    pub sections: Vec<NumericsSection>,
}

#[derive(Clone, Debug)]
pub struct NumericsEntry {
    pub key: String,
    pub value: Vec<String>,
}

pub fn read_fv_schemes(case_dir: &Path) -> Result<Option<FvSchemes>> {
    let path = case_dir.join("system").join("fvSchemes");
    if !path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(&path).map_err(|error| {
        MeshError::InvalidInput(format!("could not read {} ({error})", path.display()))
    })?;
    Ok(Some(FvSchemes {
        path: path.clone(),
        sections: parse_numerics_dictionary_str(&content, &path)?,
    }))
}

pub fn read_fv_solution(case_dir: &Path) -> Result<Option<FvSolution>> {
    let path = case_dir.join("system").join("fvSolution");
    if !path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(&path).map_err(|error| {
        MeshError::InvalidInput(format!("could not read {} ({error})", path.display()))
    })?;
    Ok(Some(FvSolution {
        path: path.clone(),
        sections: parse_numerics_dictionary_str(&content, &path)?,
    }))
}

pub fn format_numerics_value(value: &[String]) -> String {
    if value.is_empty() {
        return "empty".to_string();
    }
    value.join(" ")
}

fn parse_numerics_dictionary_str(content: &str, path: &Path) -> Result<Vec<NumericsSection>> {
    let tokens = tokenize(content);
    let mut cursor = TokenCursor::new(path, tokens);
    let mut sections = Vec::new();

    while let Some(token) = cursor.peek() {
        if token == ";" {
            cursor.next_required()?;
            continue;
        }
        if token == "FoamFile" {
            cursor.next_required()?;
            cursor.skip_braced_block()?;
            continue;
        }

        let name = cursor.next_required()?;
        if cursor.peek() == Some("{") {
            sections.push(parse_section(&mut cursor, name)?);
        } else {
            cursor.skip_value_or_block()?;
        }
    }

    Ok(sections)
}

fn parse_section(cursor: &mut TokenCursor, name: String) -> Result<NumericsSection> {
    cursor.expect("{")?;
    let mut entries = Vec::new();
    let mut sections = Vec::new();

    while !cursor.peek_is("}")? {
        if cursor.peek() == Some(";") {
            cursor.next_required()?;
            continue;
        }

        let key = cursor.next_required()?;
        if cursor.peek() == Some("{") {
            sections.push(parse_section(cursor, key)?);
            continue;
        }

        entries.push(NumericsEntry {
            key,
            value: cursor.read_value_until_semicolon()?,
        });
    }
    cursor.expect("}")?;
    cursor.expect_optional(";")?;

    Ok(NumericsSection {
        name,
        entries,
        sections,
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{format_numerics_value, parse_numerics_dictionary_str};

    #[test]
    fn parses_fv_schemes_sections() {
        let content = r#"
        FoamFile
        {
            version 2.0;
            class dictionary;
            object fvSchemes;
        }

        ddtSchemes { default Euler; }
        gradSchemes { default Gauss linear; grad(U) Gauss linear; }
        divSchemes { default none; }
        "#;

        let sections = parse_numerics_dictionary_str(content, Path::new("fvSchemes")).unwrap();

        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].name, "ddtSchemes");
        assert_eq!(sections[0].entries[0].key, "default");
        assert_eq!(
            format_numerics_value(&sections[1].entries[1].value),
            "Gauss linear"
        );
    }

    #[test]
    fn parses_nested_fv_solution_solver_sections() {
        let content = r#"
        solvers
        {
            p
            {
                solver PCG;
                preconditioner DIC;
                tolerance 1e-08;
                relTol 0.05;
            }
            U
            {
                solver smoothSolver;
            }
        }
        SIMPLE
        {
            nNonOrthogonalCorrectors 1;
        }
        "#;

        let sections = parse_numerics_dictionary_str(content, Path::new("fvSolution")).unwrap();

        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].name, "solvers");
        assert_eq!(sections[0].sections.len(), 2);
        assert_eq!(sections[0].sections[0].entries[0].key, "solver");
        assert_eq!(
            format_numerics_value(&sections[0].sections[0].entries[0].value),
            "PCG"
        );
        assert_eq!(sections[1].entries[0].key, "nNonOrthogonalCorrectors");
    }

    #[test]
    fn accepts_empty_sections() {
        let sections =
            parse_numerics_dictionary_str("solvers { }", Path::new("fvSolution")).unwrap();

        assert_eq!(sections.len(), 1);
        assert!(sections[0].entries.is_empty());
        assert!(sections[0].sections.is_empty());
    }
}
