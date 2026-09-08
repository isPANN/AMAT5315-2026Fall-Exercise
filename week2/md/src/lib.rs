pub fn greeting() -> &'static str {
    "Hello, world!"
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
