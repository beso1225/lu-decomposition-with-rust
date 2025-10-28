#[derive(Debug)]
struct Matrix {
    n: usize, // number of rows
    m: usize, // number of columns
    data: Vec<Vec<f64>>, // 2D vector to hold matrix data
    lu: bool,
}

impl Matrix {
    fn new(n: usize, m: usize) -> Self {
        Matrix {
            n,
            m,
            data: vec![vec![0.0; m]; n],
            lu: false,
        }
    }

    fn show(&self) {
        for i in 0..self.n {
            for j in 0..self.m {
                print!("{:8.4} ", self.data[i][j]);
            }
            println!();
        }
    }

    fn max_abs_in_column(&self, col: usize, start_row: usize) -> usize {
        let mut max_row = start_row;
        let mut max_value = self.data[start_row][col].abs();
        for i in (start_row + 1)..self.n {
            if self.data[i][col].abs() > max_value {
                max_value = self.data[i][col].abs();
                max_row = i;
            }
        }
        max_row
    }

    fn read_from_csv_with_right_hand_side(filename: &str) -> (Matrix, Vec<f64>) {
        /*
        Reads matrix data from a CSV file. The CSV file should have the following format:
        n, m,
        a11, a12, ..., a1m,
        a21, a22, ..., a2m,
        ...
        an1, an2, ..., anm,
        k, // number of right-hand side vectors, as usual, k = n, but if k = 0, no right-hand side vector is read
        b1, b2, ..., bk,
        */

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .trim(csv::Trim::All)
            .from_path(filename)
            .expect("Cannot open file");

        let mut vals: Vec<f64> = Vec::new();
        for result in reader.records() {
            let record = result.expect("Error reading csv record");
            for field in record.iter() {
                if field.is_empty() {
                    continue;
                }
                if let Ok(num) = field.trim().parse::<f64>() {
                    vals.push(num);
                }
            }
        }
        let n = vals[0] as usize;
        let m = vals[1] as usize;
        let mut mat = Matrix::new(n, m);
        let mut index = 2;
        for i in 0..n {
            for j in 0..m {
                mat.data[i][j] = vals[index];
                index += 1;
            }
        }
        let k = vals[index] as usize;
        if k == 0 {
            return (mat, Vec::new());
        }
        let mut b: Vec<f64> = Vec::with_capacity(k);
        index += 1;
        for _ in 0..k {
            b.push(vals[index]);
            index += 1;
        }
        (mat, b)
    }
}

fn lu_decomposition(mat: &mut Matrix) -> Vec<usize> { // Changes original matrix to L and U matrix and returns permutation vector
    if mat.n != mat.m {
        panic!("Matrix must be square for LU decomposition");
    }
    let mut p: Vec<usize> = (0..mat.n).collect();
    for k in 0..(mat.n - 1) {
        let max_row = mat.max_abs_in_column(k, k);
        if max_row != k {
            mat.data.swap(k, max_row);
            p.swap(k, max_row);
        }
        let pivot_value_inverse = 1.0 / mat.data[k][k];
        for i in (k + 1)..mat.n {
            mat.data[i][k] *= pivot_value_inverse;
            for j in (k + 1)..mat.m {
                mat.data[i][j] -= mat.data[i][k] * mat.data[k][j];
            }
        }
        // println!("After processing column {}:", k);
        // mat.show();
    }
    mat.lu = true;
    p
}

fn solve_linear_system(mat: &Matrix, p: &Vec<usize>, b: &Vec<f64>) -> Vec<f64> {
    if !mat.lu {
        panic!("Matrix must be LU decomposed to solve linear systems");
    }

    if b.len() != mat.n {
        panic!("Right-hand side vector length must match matrix size");
    }
    let mut x = vec![0.0; mat.n];
    let mut y = vec![0.0; mat.n];

    // Apply permutation to b
    for i in 0..mat.n {
        y[i] = b[p[i]];
    }
    // Forward substitution to solve Ly = Pb
    for i in 0..mat.n {
        for j in 0..i {
            y[i] -= mat.data[i][j] * y[j];
        }
    }

    // Backward substitution to solve Ux = y
    for i in (0..mat.n).rev() {
        x[i] = y[i];
        for j in (i + 1)..mat.n {
            x[i] -= mat.data[i][j] * x[j];
        }
        x[i] /= mat.data[i][i];
    }

    x
}

fn main() {
    // let mut mat = Matrix::new(3, 3);
    // mat.data = vec![
    //     vec![2.0, 1.0, 1.0],
    //     vec![4.0, -6.0, 0.0],
    //     vec![-2.0, 7.0, 2.0],
    // ];
    let (mut mat, b) = Matrix::read_from_csv_with_right_hand_side("src/test/input/input2.csv");
    println!("Original Matrix:");
    mat.show();
    println!("Right-hand side Vector b: {:?}", b);
    let p = lu_decomposition(&mut mat);
    println!("LU Decomposed Matrix:");
    mat.show();
    println!("Permutation Vector: {:?}", p);
    // let b = vec![5.0, -2.0, 9.0];
    if b.is_empty() {
        println!("No right-hand side vector provided.");
        return;
    }
    let x = solve_linear_system(&mat, &p, &b);
    println!("Solution Vector x: {:?}", x);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output_vector(filename: &str) -> Vec<f64> {
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
        vals
    }

    #[test]
    fn test_solve_10x10() {
        let (mut mat, b) = Matrix::read_from_csv_with_right_hand_side("src/test/input/input2.csv");
        let p = lu_decomposition(&mut mat);
        let x = solve_linear_system(&mat, &p, &b);
        let expected_x = output_vector("src/test/output/output2.csv");
        for i in 0..x.len() {
            assert!((x[i] - expected_x[i]).abs() < 1e-6, "Mismatch at index {}: expected {}, got {}", i, expected_x[i], x[i]);
        }
    }

    #[test]
    fn test_solve_50x50() {
        let (mut mat, b) = Matrix::read_from_csv_with_right_hand_side("src/test/input/input3.csv");
        let p = lu_decomposition(&mut mat);
        let x = solve_linear_system(&mat, &p, &b);
        let expected_x = output_vector("src/test/output/output3.csv");
        for i in 0..x.len() {
            assert!((x[i] - expected_x[i]).abs() < 1e-6, "Mismatch at index {}: expected {}, got {}", i, expected_x[i], x[i]);
        }
    }
}
