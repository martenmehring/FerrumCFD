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

#[derive(Debug)]
pub struct NumericsValidation {
    pub warnings: Vec<String>,
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

pub fn validate_fv_schemes(schemes: &FvSchemes) -> NumericsValidation {
    const REQUIRED_DEFAULT_SECTIONS: &[&str] = &[
        "ddtSchemes",
        "gradSchemes",
        "divSchemes",
        "laplacianSchemes",
        "interpolationSchemes",
        "snGradSchemes",
    ];

    let mut warnings = Vec::new();
    for section_name in REQUIRED_DEFAULT_SECTIONS {
        let Some(section) = top_level_section(&schemes.sections, section_name) else {
            warnings.push(format!("missing section '{section_name}'"));
            continue;
        };

        if !section.entries.iter().any(|entry| entry.key == "default") {
            warnings.push(format!("section '{section_name}' has no default entry"));
        }
    }

    NumericsValidation { warnings }
}

pub fn validate_fv_solution(solution: &FvSolution, field_names: &[String]) -> NumericsValidation {
    let mut warnings = Vec::new();
    let Some(solvers) = top_level_section(&solution.sections, "solvers") else {
        if !field_names.is_empty() {
            warnings.push(format!(
                "missing 'solvers' section for {} initial field(s)",
                field_names.len()
            ));
        }
        return NumericsValidation { warnings };
    };

    for solver_section in &solvers.sections {
        if !solver_section
            .entries
            .iter()
            .any(|entry| entry.key == "solver")
        {
            warnings.push(format!(
                "solvers.{} has no solver entry",
                solver_section.name
            ));
        }
    }

    let has_default_solver = solvers
        .sections
        .iter()
        .any(|section| section.name == "default");
    if !has_default_solver {
        for field_name in field_names {
            if !solvers
                .sections
                .iter()
                .any(|section| solver_section_matches_field(&section.name, field_name))
            {
                warnings.push(format!(
                    "initial field '{field_name}' has no fvSolution solver entry"
                ));
            }
        }
    }

    NumericsValidation { warnings }
}

fn top_level_section<'a>(
    sections: &'a [NumericsSection],
    name: &str,
) -> Option<&'a NumericsSection> {
    sections.iter().find(|section| section.name == name)
}

fn solver_section_matches_field(section_name: &str, field_name: &str) -> bool {
    if section_name == field_name {
        return true;
    }

    let Some(pattern) = section_name
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
    else {
        return false;
    };

    pattern
        .split('|')
        .map(str::trim)
        .any(|candidate| candidate == field_name || candidate == ".*")
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

    use super::{
        FvSchemes, FvSolution, format_numerics_value, parse_numerics_dictionary_str,
        validate_fv_schemes, validate_fv_solution,
    };

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

    #[test]
    fn warns_for_missing_scheme_defaults() {
        let sections = parse_numerics_dictionary_str(
            r#"
            ddtSchemes { Euler; }
            gradSchemes { default Gauss linear; }
            "#,
            Path::new("fvSchemes"),
        )
        .unwrap();
        let schemes = FvSchemes {
            path: Path::new("fvSchemes").to_path_buf(),
            sections,
        };

        let validation = validate_fv_schemes(&schemes);

        assert!(
            validation
                .warnings
                .iter()
                .any(|warning| warning.contains("ddtSchemes"))
        );
        assert!(
            validation
                .warnings
                .iter()
                .any(|warning| warning.contains("divSchemes"))
        );
    }

    #[test]
    fn validates_fv_solution_field_solver_entries() {
        let sections = parse_numerics_dictionary_str(
            r#"
            solvers
            {
                p
                {
                    solver PCG;
                }
                U
                {
                    tolerance 1e-08;
                }
            }
            "#,
            Path::new("fvSolution"),
        )
        .unwrap();
        let solution = FvSolution {
            path: Path::new("fvSolution").to_path_buf(),
            sections,
        };

        let validation = validate_fv_solution(
            &solution,
            &["p".to_string(), "U".to_string(), "T".to_string()],
        );

        assert!(
            validation
                .warnings
                .iter()
                .any(|warning| warning.contains("solvers.U has no solver entry"))
        );
        assert!(
            validation
                .warnings
                .iter()
                .any(|warning| warning.contains("initial field 'T'"))
        );
    }

    #[test]
    fn accepts_default_fv_solution_solver_for_fields() {
        let sections = parse_numerics_dictionary_str(
            r#"
            solvers
            {
                default
                {
                    solver smoothSolver;
                }
            }
            "#,
            Path::new("fvSolution"),
        )
        .unwrap();
        let solution = FvSolution {
            path: Path::new("fvSolution").to_path_buf(),
            sections,
        };

        let validation = validate_fv_solution(&solution, &["p".to_string(), "U".to_string()]);

        assert!(validation.warnings.is_empty());
    }

    #[test]
    fn accepts_parenthesized_field_group_solver() {
        let sections = parse_numerics_dictionary_str(
            r#"
            solvers
            {
                "(p|U)"
                {
                    solver smoothSolver;
                }
            }
            "#,
            Path::new("fvSolution"),
        )
        .unwrap();
        let solution = FvSolution {
            path: Path::new("fvSolution").to_path_buf(),
            sections,
        };

        let validation = validate_fv_solution(&solution, &["p".to_string(), "U".to_string()]);

        assert!(validation.warnings.is_empty());
    }
}
