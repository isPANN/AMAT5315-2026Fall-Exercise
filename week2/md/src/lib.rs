pub fn greeting() -> &'static str {
    "Hello, world!"
}

pub fn lennard_jones_energy(r: f64) -> f64 {
    4.0 * (r.powi(-12) - r.powi(-6))
}

pub fn lennard_jones_force(r: f64) -> f64 {
    24.0 * (2.0 * r.powi(-13) - r.powi(-7))
}

#[cfg(test)]
mod tests {
    use super::greeting;

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }

    #[test]
    fn lennard_jones_energy_at_two() {
        assert_eq!(super::lennard_jones_energy(2.0), -0.0615234375);
    }

    #[test]
    fn lennard_jones_force_at_two() {
        assert_eq!(super::lennard_jones_force(2.0), -0.181640625);
    }
}
