#[derive(Debug, Clone)]
pub enum Matrix {
    Dense(DenseMatrix),
    Sparse(SparseMatrix),
}

#[allow(dead_code)]
impl Matrix {
    pub fn new(n: usize, m: usize, dense: bool) -> Self {
        if dense {
            Matrix::Dense(DenseMatrix::new(n, m))
        } else {
            Matrix::Sparse(SparseMatrix::new(n, m))
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        match self {
            Matrix::Dense(dense) => dense.get(row, col),
            Matrix::Sparse(sparse) => sparse.get(row, col),
        }
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        match self {
            Matrix::Dense(dense) => dense.set(row, col, value),
            Matrix::Sparse(sparse) => sparse.set(row, col, value),
        }
    }

    pub fn change(&mut self) {
        match self {
            Matrix::Dense(dense) => {
                *self = Matrix::Sparse(SparseMatrix::from_dense(dense));
            },
            Matrix::Sparse(sparse) => {
                *self = Matrix::Dense(DenseMatrix::from_sparse(sparse));
            },
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Matrix::Dense(dense) => dense.is_empty(),
            Matrix::Sparse(sparse) => sparse.is_empty(),
        }
    }

    pub fn is_symmetric_positive_definite(&self) -> bool {
        match self {
            Matrix::Dense(dense) => dense.is_symmetric_positive_definite(),
            Matrix::Sparse(sparse) => sparse.is_symmetric_positive_definite(),
        }
    }

    pub fn norm_inf(&self) -> f64 {
        match self {
            Matrix::Dense(dense) => dense.norm_inf(),
            Matrix::Sparse(sparse) => sparse.norm_inf(),
        }
    }

    pub fn copy(&self) -> Matrix {
        match self {
            Matrix::Dense(dense) => Matrix::Dense(dense.copy()),
            Matrix::Sparse(sparse) => Matrix::Sparse(sparse.copy()),
        }
    }

    pub fn scalar_multiply(&self, scalar: f64) -> Matrix {
        match self {
            Matrix::Dense(dense) => Matrix::Dense(dense.scalar_multiply(scalar)),
            Matrix::Sparse(sparse) => Matrix::Sparse(sparse.scalar_multiply(scalar)),
        }
    }

    pub fn plus(left: &Matrix, right: &Matrix) -> Matrix {
        match (left, right) {
            (Matrix::Dense(ld), Matrix::Dense(rd)) => Matrix::Dense(DenseMatrix::plus(ld, rd)),
            (Matrix::Sparse(ls), Matrix::Sparse(rs)) => Matrix::Sparse(SparseMatrix::plus(ls, rs)),
            _ => panic!("Matrix types must match for addition"),
        }
    }

    pub fn minus(left: &Matrix, right: &Matrix) -> Matrix {
        match (left, right) {
            (Matrix::Dense(ld), Matrix::Dense(rd)) => Matrix::Dense(DenseMatrix::minus(ld, rd)),
            (Matrix::Sparse(ls), Matrix::Sparse(rs)) => Matrix::Sparse(SparseMatrix::minus(ls, rs)),
            _ => panic!("Matrix types must match for subtraction"),
        }
    }

    pub fn product(left: &Matrix, right: &Matrix) -> Matrix {
        match (left, right) {
            (Matrix::Dense(ld), Matrix::Dense(rd)) => Matrix::Dense(DenseMatrix::product(ld, rd)),
            (Matrix::Sparse(ls), Matrix::Sparse(rs)) => Matrix::Sparse(SparseMatrix::product(ls, rs)),
            _ => panic!("Matrix types must match for multiplication"),
        }
    }

    pub fn inner_product(vec1: &Matrix, vec2: &Matrix) -> f64 {
        match (vec1, vec2) {
            (Matrix::Dense(v1), Matrix::Dense(v2)) => DenseMatrix::inner_product(v1, v2),
            (Matrix::Sparse(v1), Matrix::Sparse(v2)) => SparseMatrix::inner_product(v1, v2),
            _ => panic!("Both inputs must be of the same matrix type for inner product"),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum MatrixState {
    Original,
    Lu,
    Cc,
}

#[derive(Debug, Clone)]
pub struct DenseMatrix {
    pub n: usize, // number of rows
    pub m: usize, // number of columns
    pub data: Vec<Vec<f64>>, // 2D vector to hold matrix data
    pub state: MatrixState,
}


#[allow(dead_code)]
impl DenseMatrix {
    pub fn new(n: usize, m: usize) -> Self {
        DenseMatrix {
            n,
            m,
            data: vec![vec![0.0; m]; n],
            state: MatrixState::Original,
        }
    }

    pub fn show(&self) {
        for i in 0..self.n {
            for j in 0..self.m {
                print!("{:8.4} ", self.data[i][j]);
            }
            println!();
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row][col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row][col] = value;
    }

    pub fn copy(&self) -> DenseMatrix {
        let mut new_mat = DenseMatrix::new(self.n, self.m);
        for i in 0..self.n {
            for j in 0..self.m {
                new_mat.data[i][j] = self.data[i][j];
            }
        }
        new_mat
    }

    pub fn norm_inf(&self) -> f64 {
        let mut max_sum = 0.0;
        for i in 0..self.n {
            let row_sum: f64 = self.data[i].iter().map(|&x| x.abs()).sum();
            if row_sum > max_sum {
                max_sum = row_sum;
            }
        }
        max_sum
    }

    pub fn from_sparse(sparse: &SparseMatrix) -> DenseMatrix {
        // Convert SparseMatrix to DenseMatrix
        unimplemented!()
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0 || self.m == 0
    }

    pub fn is_symmetric_positive_definite(&self) -> bool {
        if self.n != self.m {
            return false;
        }
        for i in 0..self.n {
            for j in 0..self.m {
                if self.data[i][j] != self.data[j][i] {
                    return false;
                }
            }
        }
        // Note: This only checks symmetry. Positive definiteness check is more complex and not implemented here.
        true
    }

    // pub fn product_vector_left(&self, vec: &Vec<f64>) -> Vec<f64> {
    //     if vec.len() != self.n {
    //         panic!("Vector length must match number of matrix rows");
    //     }
    //     let mut result = vec![0.0; self.m];
    //     for j in 0..self.m {
    //         for i in 0..self.n {
    //             result[j] += vec[i] * self.data[i][j];
    //         }
    //     }
    //     result
    // }

    // pub fn product_vector_right(&self, vec: &Vec<f64>) -> Vec<f64> {
    //     if vec.len() != self.m {
    //         panic!("Vector length must match number of matrix columns");
    //     }
    //     let mut result = vec![0.0; self.n];
    //     for i in 0..self.n {
    //         for j in 0..self.m {
    //             result[i] += self.data[i][j] * vec[j];
    //         }
    //     }
    //     result
    // }

    pub fn plus(left: &DenseMatrix, right: &DenseMatrix) -> DenseMatrix {
        if left.n != right.n || left.m != right.m {
            panic!("Matrices must have the same dimensions for addition");
        }
        let mut result = DenseMatrix::new(left.n, left.m);
        for i in 0..left.n {
            for j in 0..left.m {
                result.data[i][j] = left.data[i][j] + right.data[i][j];
            }
        }
        result
    }

    pub fn minus(left: &DenseMatrix, right: &DenseMatrix) -> DenseMatrix {
        if left.n != right.n || left.m != right.m {
            panic!("Matrices must have the same dimensions for subtraction");
        }
        let mut result = DenseMatrix::new(left.n, left.m);
        for i in 0..left.n {
            for j in 0..left.m {
                result.data[i][j] = left.data[i][j] - right.data[i][j];
            }
        }
        result
    }

    pub fn product(left: &DenseMatrix, right: &DenseMatrix) -> DenseMatrix {
        if left.m != right.n {
            panic!("Incompatible matrix dimensions for multiplication");
        }
        let mut result = DenseMatrix::new(left.n, right.m);
        for i in 0..left.n {
            for j in 0..right.m {
                for k in 0..left.m {
                    result.data[i][j] += left.data[i][k] * right.data[k][j];
                }
            }
        }
        result
    }

    pub fn inner_product(vec1: &DenseMatrix, vec2: &DenseMatrix) -> f64 {
        if vec1.n != vec2.n || vec1.m != 1 || vec2.m != 1 {
            panic!("Both inputs must be column vectors of the same length for inner product");
        }
        let mut result = 0.0;
        for i in 0..vec1.n {
            result += vec1.data[i][0] * vec2.data[i][0];
        }
        result
    }

    pub fn scalar_multiply(&self, scalar: f64) -> DenseMatrix {
        let mut result = DenseMatrix::new(self.n, self.m);
        for i in 0..self.n {
            for j in 0..self.m {
                result.data[i][j] = self.data[i][j] * scalar;
            }
        }
        result
    }

    pub fn max_abs_in_column(&self, col: usize, start_row: usize) -> usize {
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

    pub fn read_from_csv_with_right_hand_side(filename: &str) -> (DenseMatrix, DenseMatrix) {
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
        let mut mat = DenseMatrix::new(n, m);
        let mut index = 2;
        for i in 0..n {
            for j in 0..m {
                mat.data[i][j] = vals[index];
                index += 1;
            }
        }
        let k = vals[index] as usize;
        if k == 0 {
            return (mat, DenseMatrix::new(0, 0));
        }
        let mut b = DenseMatrix::new(k, 1);
        index += 1;
        for i in 0..k {
            b.data[i][0] = vals[index];
            index += 1;
        }
        (mat, b)
    }
}

#[derive(Debug, Clone)]
pub struct SparseMatrix {
    // Sparse matrix representation (e.g., CSR, CSC) can be defined here
}

impl SparseMatrix {
    pub fn new(n: usize, m: usize) -> Self {
        SparseMatrix {
            // Initialize sparse matrix representation here
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        // Implement getting value from sparse matrix
        unimplemented!()
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        // Implement setting value in sparse matrix
        unimplemented!()
    }

    pub fn from_dense(dense: &DenseMatrix) -> SparseMatrix {
        // Convert DenseMatrix to SparseMatrix
        unimplemented!()
    }

    pub fn is_empty(&self) -> bool {
        // Implement check for empty sparse matrix
        unimplemented!()
    }

    pub fn copy(&self) -> SparseMatrix {
        // Implement deep copy for sparse matrix
        unimplemented!()
    }

    pub fn norm_inf(&self) -> f64 {
        // Implement infinity norm calculation for sparse matrix
        unimplemented!()
    }

    pub fn is_symmetric_positive_definite(&self) -> bool {
        // Implement check for symmetric positive definiteness for sparse matrix
        unimplemented!()
    }

    pub fn scalar_multiply(&self, scalar: f64) -> SparseMatrix {
        // Implement scalar multiplication for sparse matrix
        unimplemented!()
    }

    pub fn plus(left: &SparseMatrix, right: &SparseMatrix) -> SparseMatrix {
        // Implement addition for sparse matrices
        unimplemented!()
    }

    pub fn minus(left: &SparseMatrix, right: &SparseMatrix) -> SparseMatrix {
        // Implement subtraction for sparse matrices
        unimplemented!()
    }

    pub fn product(left: &SparseMatrix, right: &SparseMatrix) -> SparseMatrix {
        // Implement multiplication for sparse matrices
        unimplemented!()
    }

    pub fn inner_product(vec1: &SparseMatrix, vec2: &SparseMatrix) -> f64 {
        // Implement inner product for sparse vectors
        unimplemented!()
    }
}