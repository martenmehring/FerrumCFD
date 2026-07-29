//! Solver-independent normalization of steady-state continuity errors.

use ferrum_mesh::flow::ContinuitySummary;
use ferrum_mesh::{MeshError, Result};

/// OpenFOAM Foundation-style steady continuity errors.
///
/// Both values are dimensionless when the raw continuity sums, cell volume,
/// and pseudo-time step use consistent units. `global` preserves the sign of
/// the raw global imbalance.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NormalizedContinuity {
    pub local: f64,
    pub global: f64,
}

/// Normalizes raw steady-state continuity sums using `continuityErrs`
/// semantics.
///
/// The local error is `(sum(abs(div(phi))) / total_cell_volume) * delta_t`; the
/// global error is `(sum(div(phi)) / total_cell_volume) * delta_t`. The
/// volume-weighted average is evaluated before the pseudo-time scaling, as in
/// the Foundation implementation.
pub fn normalize_steady_continuity(
    raw: ContinuitySummary,
    total_cell_volume: f64,
    delta_t: f64,
) -> Result<NormalizedContinuity> {
    validate_raw(raw)?;
    validate_positive_finite(total_cell_volume, "total cell volume")?;
    validate_positive_finite(delta_t, "deltaT")?;

    let local = (raw.sum_abs / total_cell_volume) * delta_t;
    let global = (raw.global_sum / total_cell_volume) * delta_t;
    if !local.is_finite() || !global.is_finite() {
        return Err(invalid(
            "normalized steady continuity errors must be finite",
        ));
    }

    Ok(NormalizedContinuity { local, global })
}

fn validate_raw(raw: ContinuitySummary) -> Result<()> {
    for (name, value) in [
        ("l2 norm", raw.l2_norm),
        ("maximum absolute error", raw.max_abs),
        ("sum of absolute errors", raw.sum_abs),
        ("global sum", raw.global_sum),
    ] {
        if !value.is_finite() {
            return Err(invalid(format!(
                "raw steady continuity {name} must be finite"
            )));
        }
    }

    for (name, value) in [
        ("l2 norm", raw.l2_norm),
        ("maximum absolute error", raw.max_abs),
        ("sum of absolute errors", raw.sum_abs),
    ] {
        if value < 0.0 {
            return Err(invalid(format!(
                "raw steady continuity {name} must be non-negative"
            )));
        }
    }

    Ok(())
}

fn validate_positive_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(format!("{name} must be positive and finite")));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> MeshError {
    MeshError::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(l2_norm: f64, max_abs: f64, sum_abs: f64, global_sum: f64) -> ContinuitySummary {
        ContinuitySummary {
            l2_norm,
            max_abs,
            sum_abs,
            global_sum,
        }
    }

    fn assert_invalid(result: Result<NormalizedContinuity>) {
        assert!(matches!(result, Err(MeshError::InvalidInput(_))));
    }

    #[test]
    fn zero_raw_continuity_normalizes_to_zero() {
        let normalized = normalize_steady_continuity(raw(0.0, 0.0, 0.0, 0.0), 3.0, 2.0)
            .expect("zero continuity must normalize");

        assert_eq!(normalized, NormalizedContinuity::default());
    }

    #[test]
    fn global_sign_is_preserved() {
        let positive = normalize_steady_continuity(raw(1.0, 1.0, 8.0, 3.0), 4.0, 2.0)
            .expect("positive global sum must normalize");
        let negative = normalize_steady_continuity(raw(1.0, 1.0, 8.0, -3.0), 4.0, 2.0)
            .expect("negative global sum must normalize");

        assert_eq!(positive.local, 4.0);
        assert_eq!(positive.global, 1.5);
        assert_eq!(negative.local, 4.0);
        assert_eq!(negative.global, -1.5);
        assert!(negative.global.is_sign_negative());
    }

    #[test]
    fn normalization_scales_linearly_with_delta_t_and_inverse_volume() {
        let baseline = normalize_steady_continuity(raw(2.0, 1.0, 12.0, -4.0), 6.0, 0.5)
            .expect("baseline continuity must normalize");
        let doubled_delta_t = normalize_steady_continuity(raw(2.0, 1.0, 12.0, -4.0), 6.0, 1.0)
            .expect("scaled deltaT must normalize");
        let doubled_volume = normalize_steady_continuity(raw(2.0, 1.0, 12.0, -4.0), 12.0, 0.5)
            .expect("scaled volume must normalize");

        assert_eq!(baseline.local, 1.0);
        assert_eq!(baseline.global, -1.0 / 3.0);
        assert_eq!(doubled_delta_t.local, 2.0 * baseline.local);
        assert_eq!(doubled_delta_t.global, 2.0 * baseline.global);
        assert_eq!(doubled_volume.local, 0.5 * baseline.local);
        assert_eq!(doubled_volume.global, 0.5 * baseline.global);
    }

    #[test]
    fn every_non_finite_raw_member_is_rejected() {
        let finite = [1.0, 1.0, 1.0, 0.0];
        for index in 0..finite.len() {
            for invalid_value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut values = finite;
                values[index] = invalid_value;
                assert_invalid(normalize_steady_continuity(
                    raw(values[0], values[1], values[2], values[3]),
                    1.0,
                    1.0,
                ));
            }
        }
    }

    #[test]
    fn every_non_negative_raw_member_rejects_negative_values() {
        for index in 0..3 {
            let mut values = [1.0, 1.0, 1.0, -1.0];
            values[index] = -f64::MIN_POSITIVE;
            assert_invalid(normalize_steady_continuity(
                raw(values[0], values[1], values[2], values[3]),
                1.0,
                1.0,
            ));
        }
    }

    #[test]
    fn invalid_total_cell_volume_is_rejected() {
        for invalid_value in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_invalid(normalize_steady_continuity(
                raw(1.0, 1.0, 1.0, 0.0),
                invalid_value,
                1.0,
            ));
        }
    }

    #[test]
    fn invalid_delta_t_is_rejected() {
        for invalid_value in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_invalid(normalize_steady_continuity(
                raw(1.0, 1.0, 1.0, 0.0),
                1.0,
                invalid_value,
            ));
        }
    }

    #[test]
    fn overflowing_normalized_results_are_rejected() {
        assert_invalid(normalize_steady_continuity(
            raw(1.0, 1.0, f64::MAX, 0.0),
            1.0,
            2.0,
        ));
        assert_invalid(normalize_steady_continuity(
            raw(1.0, 1.0, 0.0, f64::MAX),
            1.0,
            2.0,
        ));
        assert_invalid(normalize_steady_continuity(
            raw(1.0, 1.0, 0.0, -f64::MAX),
            1.0,
            2.0,
        ));
    }

    #[test]
    fn weighted_average_is_evaluated_before_delta_t_scaling() {
        let normalized = normalize_steady_continuity(raw(1.0, 1.0, f64::MAX, -f64::MAX), 2.0, 2.0)
            .expect("finite weighted averages must not overflow before division");

        assert_eq!(normalized.local, f64::MAX);
        assert_eq!(normalized.global, -f64::MAX);
    }
}
