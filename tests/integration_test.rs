#[cfg(test)]
mod tests {
    use lu::solutions::{lu_solve, cg};
    use lu::matrix::Matrix;

    fn output_vector(filename: &str) -> Matrix {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .trim(csv::Trim::All)
            .from_path(filename)
            .expect("Cannot open output CSV file");

        let mut vals: Vec<f64> = Vec::new();
        for result in reader.records() {
            let record = result.expect("Error reading csv record");
            for field in record.iter() {
                let s = field.trim();
                if s.is_empty() { continue; }
                if let Ok(num) = s.parse::<f64>() {
                    vals.push(num);
                } else {
                    panic!("Failed to parse '{}' as f64 in {}", s, filename);
                }
            }
        }
        let n = vals.len();
        let mut matrix = Matrix::new(n, 1, true);
        for i in 0..n {
            matrix.set(i, 0, vals[i]);
        }
        matrix
    }

    #[test]
    fn test_solve_10x10() {
        let (mut mat, b) = Matrix::read_from_csv_with_right_hand_side("tests/input/input2.csv", true);
        let p = mat.lu_decomposition();
        let x = lu_solve::solve(&mat, &p, &b);
        let expected_x = output_vector("tests/output/output2.csv");
        for i in 0..x.n() {
            assert!((x.get(i, 0) - expected_x.get(i, 0)).abs() < 1e-6, "Mismatch at index {}: expected {}, got {}", i, expected_x.get(i, 0), x.get(i, 0));
        }
    }

    #[test]
    fn test_solve_50x50() {
        let (mut mat, b) = Matrix::read_from_csv_with_right_hand_side("tests/input/input3.csv", true);
        let p = mat.lu_decomposition();
        let x = lu_solve::solve(&mat, &p, &b);
        let expected_x = output_vector("tests/output/output3.csv");
        for i in 0..x.n() {
            assert!((x.get(i, 0) - expected_x.get(i, 0)).abs() < 1e-6, "Mismatch at index {}: expected {}, got {}", i, expected_x.get(i, 0), x.get(i, 0));
        }
    }

    #[test]
    fn test_solve_50x50_with_cg() {
        let (mat, b) = Matrix::read_from_csv_with_right_hand_side("tests/input/input3.csv", true);
        let initial_guess = None;
        let max_iter = 1000;
        let x = cg::solve(&mat, &b, initial_guess, max_iter);
        let expected_x = output_vector("tests/output/output3.csv");
        for i in 0..x.n() {
            assert!((x.get(i, 0) - expected_x.get(i, 0)).abs() < 1e-6, "Mismatch at index {}: expected {}, got {}", i, expected_x.get(i, 0), x.get(i, 0));
        }
    }
}