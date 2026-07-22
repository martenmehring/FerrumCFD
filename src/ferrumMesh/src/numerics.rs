use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::dictionary::{MAX_DICTIONARY_NESTING, Token, TokenCursor, TokenProvenance, tokenize};
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
        let mut configured_fields = HashSet::new();
        let mut wildcard = false;
        for section in &solvers.sections {
            if let Some(pattern) = section
                .name
                .strip_prefix('(')
                .and_then(|value| value.strip_suffix(')'))
            {
                for candidate in pattern.split('|').map(str::trim) {
                    if candidate == ".*" {
                        wildcard = true;
                    } else {
                        configured_fields.insert(candidate);
                    }
                }
            } else {
                configured_fields.insert(section.name.as_str());
            }
        }
        for field_name in field_names {
            if !wildcard && !configured_fields.contains(field_name.as_str()) {
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

fn parse_numerics_dictionary_str(content: &str, path: &Path) -> Result<Vec<NumericsSection>> {
    let mut cursor = tokenize(path, content)?.into_cursor();
    let mut sections = Vec::new();

    while let Some(token) = cursor.peek()? {
        if structural(token, ";") {
            cursor.next_required()?;
            continue;
        }
        if ordinary(token, "FoamFile") {
            cursor.next_required()?;
            cursor.skip_braced_block()?;
            continue;
        }

        let (name, provenance) = take_name(&mut cursor, false)?;
        if cursor.peek()?.is_some_and(|token| structural(token, "{")) {
            let role = if provenance == TokenProvenance::Ordinary && name == "solvers" {
                SectionRole::Solvers
            } else {
                SectionRole::General
            };
            sections.push(parse_section(&mut cursor, name, role, 1)?);
        } else {
            cursor.skip_value_or_block()?;
        }
    }

    Ok(sections)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum SectionRole {
    General,
    Solvers,
}

fn parse_section(
    cursor: &mut TokenCursor,
    name: String,
    role: SectionRole,
    depth: usize,
) -> Result<NumericsSection> {
    if depth > MAX_DICTIONARY_NESTING {
        return Err(MeshError::InvalidInput(format!(
            "numerics dictionary nesting exceeds {MAX_DICTIONARY_NESTING} levels in {}",
            cursor.path().display()
        )));
    }
    cursor.expect("{")?;
    let mut entries = Vec::new();
    let mut sections = Vec::new();

    while !cursor.peek()?.is_some_and(|token| structural(token, "}")) {
        if cursor.peek()?.is_some_and(|token| structural(token, ";")) {
            cursor.next_required()?;
            continue;
        }

        let (key, _) = take_name(cursor, role == SectionRole::Solvers)?;
        if cursor.peek()?.is_some_and(|token| structural(token, "{")) {
            sections.push(parse_section(cursor, key, SectionRole::General, depth + 1)?);
            continue;
        }

        entries.push(NumericsEntry {
            key,
            value: cursor.read_provenance_preserving_bare_entry()?,
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

fn structural(token: &Token, expected: &str) -> bool {
    token.provenance == TokenProvenance::Structural && token.value == expected
}

fn ordinary(token: &Token, expected: &str) -> bool {
    token.provenance == TokenProvenance::Ordinary && token.value == expected
}

fn take_name(
    cursor: &mut TokenCursor,
    allow_quoted_field_selector: bool,
) -> Result<(String, TokenProvenance)> {
    if cursor
        .peek()?
        .is_some_and(|token| token.provenance == TokenProvenance::Structural)
    {
        return cursor.reject_current_as("numerics name must not be structural punctuation");
    }
    let quoted_inert = cursor.peek()?.is_some_and(|token| {
        token.provenance == TokenProvenance::Quoted
            && !(allow_quoted_field_selector && narrow_field_selector(&token.value))
    });
    if quoted_inert {
        cursor.try_reserve_current_value(2)?;
    }
    let mut token = cursor.next_required()?;
    if quoted_inert {
        token.value.insert(0, '"');
        token.value.push('"');
    }
    Ok((token.value, token.provenance))
}

fn narrow_field_selector(value: &str) -> bool {
    let Some(inner) = value
        .strip_prefix('(')
        .and_then(|candidate| candidate.strip_suffix(')'))
    else {
        return false;
    };
    !inner.is_empty()
        && inner.split('|').all(|candidate| {
            candidate == ".*"
                || candidate
                    .strip_prefix(|ch: char| ch.is_ascii_alphabetic() || ch == '_')
                    .is_some_and(|tail| {
                        tail.chars()
                            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
                    })
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

    #[test]
    fn quoted_reserved_numerics_tokens_are_inert() {
        let sections = parse_numerics_dictionary_str(
            r#"
            "FoamFile" { object decoy; }
            "solvers" { p { solver PCG; } }
            solvers
            {
                "default" { solver smoothSolver; }
                p { "solver" PCG; }
            }
            ddtSchemes { "default" Euler; }
            "default" ignored;
            "#,
            Path::new("fvSolution"),
        )
        .unwrap();

        assert!(
            sections
                .iter()
                .any(|section| section.name == "\"FoamFile\"")
        );
        assert!(sections.iter().any(|section| section.name == "\"solvers\""));
        assert!(sections.iter().any(|section| section.name == "solvers"));
        let solution = FvSolution {
            path: Path::new("fvSolution").to_path_buf(),
            sections: sections.clone(),
        };
        let solution_validation = validate_fv_solution(&solution, &["p".into(), "T".into()]);
        assert!(
            solution_validation
                .warnings
                .iter()
                .any(|warning| warning.contains("solvers.p has no solver entry"))
        );
        assert!(
            solution_validation
                .warnings
                .iter()
                .any(|warning| warning.contains("initial field 'T'"))
        );

        let schemes = FvSchemes {
            path: Path::new("fvSchemes").to_path_buf(),
            sections,
        };
        let schemes_validation = validate_fv_schemes(&schemes);
        assert!(
            schemes_validation
                .warnings
                .iter()
                .any(|warning| warning.contains("ddtSchemes"))
        );
    }

    #[test]
    fn only_narrow_quoted_field_selector_patterns_are_unquoted() {
        let sections = parse_numerics_dictionary_str(
            r#"
            solvers
            {
                "(p|U)" { solver smoothSolver; }
                "default" { solver PCG; }
                "(p|U|*)" { solver PCG; }
            }
            "#,
            Path::new("fvSolution"),
        )
        .unwrap();
        let solvers = &sections[0];

        assert_eq!(solvers.sections[0].name, "(p|U)");
        assert_eq!(solvers.sections[1].name, "\"default\"");
        assert_eq!(solvers.sections[2].name, "\"(p|U|*)\"");
        let solution = FvSolution {
            path: Path::new("fvSolution").to_path_buf(),
            sections,
        };
        assert!(
            validate_fv_solution(&solution, &["p".into(), "U".into()])
                .warnings
                .is_empty()
        );
    }

    #[test]
    fn quoted_structural_spelling_is_not_numerics_syntax() {
        let sections = parse_numerics_dictionary_str(
            r#"
            solvers
            {
                p
                {
                    ";" value;
                    marker ";";
                    "}" inert;
                    solver PCG;
                }
            }
            "#,
            Path::new("fvSolution"),
        )
        .unwrap();
        let entries = &sections[0].sections[0].entries;

        assert_eq!(entries[0].key, "\";\"");
        assert_eq!(entries[1].value, ["\";\""]);
        assert_eq!(entries[2].key, "\"}\"");
        assert_eq!(entries[3].key, "solver");
    }

    #[test]
    fn rejects_excessive_dictionary_nesting_without_stack_overflow() {
        let mut content = String::new();
        for index in 0..=super::MAX_DICTIONARY_NESTING {
            content.push_str(&format!("section{index} {{ "));
        }
        for _ in 0..=super::MAX_DICTIONARY_NESTING {
            content.push_str("} ");
        }

        let error = parse_numerics_dictionary_str(&content, Path::new("fvSolution"))
            .expect_err("excessive nesting must fail");

        assert!(
            error
                .to_string()
                .contains("dictionary nesting limit exceeded")
        );
    }
}
