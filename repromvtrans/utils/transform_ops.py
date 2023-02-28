def scale_basis(transform_matrix, scale):
    # TODO: This seems weird.
    tf = transform_matrix.clone()
    tf[:, :3, :] *= scale
    return tf
