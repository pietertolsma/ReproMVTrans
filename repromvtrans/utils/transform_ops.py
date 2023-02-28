def scale_basis(transform_matrix, scale):
    tf = transform_matrix.clone()
    tf[:, :2, :] *= scale
    return tf
