#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <vector>
#include <exception>

namespace sjtu {

class size_mismatch : public std::exception {
public:
    const char *what() const noexcept override {
        return "Size mismatch";
    }
};

class invalid_index : public std::exception {
public:
    const char *what() const noexcept override {
        return "Index out of range";
    }
};

template <typename T>
class CSRMatrix {

private:
    size_t n_rows;
    size_t n_cols;
    size_t nnz;
    std::vector<size_t> indptr;
    std::vector<size_t> indices;
    std::vector<T> data;

public:
    // Assignment operators are deleted
    CSRMatrix &operator=(const CSRMatrix &other) = delete;
    CSRMatrix &operator=(CSRMatrix &&other) = delete;

    // Constructor for empty matrix with dimensions
    CSRMatrix(size_t n, size_t m) : n_rows(n), n_cols(m), nnz(0) {
        indptr.resize(n + 1, 0);
    }

    // Constructor with pre-built CSR components
    CSRMatrix(size_t n, size_t m, size_t count,
        const std::vector<size_t> &indptr_in,
        const std::vector<size_t> &indices_in,
        const std::vector<T> &data_in)
        : n_rows(n), n_cols(m), nnz(count),
          indptr(indptr_in), indices(indices_in), data(data_in) {
    }

    // Copy constructor
    CSRMatrix(const CSRMatrix &other) = default;

    // Move constructor
    CSRMatrix(CSRMatrix &&other) = default;

    // Constructor from dense matrix format (given as vector of vectors)
    CSRMatrix(size_t n, size_t m, const std::vector<std::vector<T>> &matrix)
        : n_rows(n), n_cols(m), nnz(0) {
        indptr.resize(n + 1, 0);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                if (matrix[i][j] != T()) {  // Non-zero element
                    indices.push_back(j);
                    data.push_back(matrix[i][j]);
                    ++nnz;
                }
            }
            indptr[i + 1] = nnz;
        }
    }

    // Destructor
    ~CSRMatrix() = default;

    // Get dimensions and non-zero count
    size_t getRowSize() const {
        return n_rows;
    }

    size_t getColSize() const {
        return n_cols;
    }

    size_t getNonZeroCount() const {
        return nnz;
    }

    // Element access
    T get(size_t i, size_t j) const {
        if (i >= n_rows || j >= n_cols) {
            throw invalid_index();
        }

        size_t row_start = indptr[i];
        size_t row_end = indptr[i + 1];

        for (size_t idx = row_start; idx < row_end; ++idx) {
            if (indices[idx] == j) {
                return data[idx];
            }
        }

        return T();  // Default constructed value if not found
    }

    // Set element at position (i,j), updating CSR structure as needed
    void set(size_t i, size_t j, const T &value) {
        if (i >= n_rows || j >= n_cols) {
            throw invalid_index();
        }

        size_t row_start = indptr[i];
        size_t row_end = indptr[i + 1];

        // Search for existing element
        for (size_t idx = row_start; idx < row_end; ++idx) {
            if (indices[idx] == j) {
                // Element exists, update it
                data[idx] = value;
                return;
            }
            if (indices[idx] > j) {
                // Insert before this position
                indices.insert(indices.begin() + idx, j);
                data.insert(data.begin() + idx, value);
                ++nnz;
                // Update indptr for subsequent rows
                for (size_t k = i + 1; k <= n_rows; ++k) {
                    ++indptr[k];
                }
                return;
            }
        }

        // Insert at end of row
        indices.insert(indices.begin() + row_end, j);
        data.insert(data.begin() + row_end, value);
        ++nnz;
        // Update indptr for subsequent rows
        for (size_t k = i + 1; k <= n_rows; ++k) {
            ++indptr[k];
        }
    }

    // Access CSR components
    const std::vector<size_t> &getIndptr() const {
        return indptr;
    }

    const std::vector<size_t> &getIndices() const {
        return indices;
    }

    const std::vector<T> &getData() const {
        return data;
    }

    // Convert to dense matrix format
    std::vector<std::vector<T>> getMatrix() const {
        std::vector<std::vector<T>> result(n_rows, std::vector<T>(n_cols, T()));

        for (size_t i = 0; i < n_rows; ++i) {
            size_t row_start = indptr[i];
            size_t row_end = indptr[i + 1];

            for (size_t idx = row_start; idx < row_end; ++idx) {
                result[i][indices[idx]] = data[idx];
            }
        }

        return result;
    }

    // Matrix-vector multiplication
    std::vector<T> operator*(const std::vector<T> &vec) const {
        if (vec.size() != n_cols) {
            throw size_mismatch();
        }

        std::vector<T> result(n_rows, T());

        for (size_t i = 0; i < n_rows; ++i) {
            size_t row_start = indptr[i];
            size_t row_end = indptr[i + 1];

            T sum = T();
            for (size_t idx = row_start; idx < row_end; ++idx) {
                sum = sum + data[idx] * vec[indices[idx]];
            }
            result[i] = sum;
        }

        return result;
    }

    // Row slicing
    CSRMatrix getRowSlice(size_t l, size_t r) const {
        if (l > r || r > n_rows) {
            throw invalid_index();
        }

        size_t new_rows = r - l;
        size_t start_idx = indptr[l];
        size_t end_idx = indptr[r];
        size_t new_nnz = end_idx - start_idx;

        std::vector<size_t> new_indptr(new_rows + 1);
        std::vector<size_t> new_indices(new_nnz);
        std::vector<T> new_data(new_nnz);

        for (size_t i = 0; i <= new_rows; ++i) {
            new_indptr[i] = indptr[l + i] - start_idx;
        }

        for (size_t i = 0; i < new_nnz; ++i) {
            new_indices[i] = indices[start_idx + i];
            new_data[i] = data[start_idx + i];
        }

        return CSRMatrix(new_rows, n_cols, new_nnz, new_indptr, new_indices, new_data);
    }
};

}

#endif // CSR_MATRIX_HPP
