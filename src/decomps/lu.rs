use crate::matrix::{DenseMatrix, MatrixState};

impl DenseMatrix {
    pub fn lu_decomposition(&mut self) -> DenseMatrix { // Changes original matrix to L and U matrix and returns permutation vector
        if self.n != self.m {
            panic!("Matrix must be square for LU decomposition");
        }
        let mut p = DenseMatrix::new(self.n, 1);
        for i in 0..self.n {
            p.data[i][0] = i as f64;
        }
        for k in 0..(self.n - 1) {
            let max_row = self.max_abs_in_column(k, k);
            if max_row != k {
                self.data.swap(k, max_row);
                p.data.swap(k, max_row);
            }
            let pivot_value_inverse = 1.0 / self.data[k][k];
            for i in (k + 1)..self.n {
                self.data[i][k] *= pivot_value_inverse;
                for j in (k + 1)..self.m {
                    self.data[i][j] -= self.data[i][k] * self.data[k][j];
                }
            }
            // println!("After processing column {}:", k);
            // mat.show();
        }
        self.state = MatrixState::Lu;
        p
    }
}
