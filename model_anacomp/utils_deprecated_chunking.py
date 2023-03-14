def _get_chunk_data(layer, row_id, i, j):
    # DEPRECATED
    # See description of _get_chunk_map()
    """
    _get_chunk_map() helper 
    """
    chunk = {}
    chunk['start'] = i
    chunk['end']   = j
    chunk['layer'] = layer
    chunk['row']   = row_id
    chunk['size'] = j - i + 1
    return chunk

def _get_chunks(layer, row_id, i, j, max_row_id, num_chunks):
    # DEPRECATED
    # See description of _get_chunk_map()
    """
    _get_chunk_map() helper 
    """
    chunks = []
  
    if num_chunks == 1:
      chunk = _get_chunk_data(layer, row_id, i, j)
      assert (chunk['row'] <= max_row_id)
      assert (chunk['start'] != chunk['end'])
      chunks.append(chunk)
      return chunks
  
    arr = np.array_split(range(i,j), num_chunks)
    for item in arr:
      chunk = _get_chunk_data(layer, row_id, i=item[0], j=item[item.size-1])
      assert (chunk['end'] <= j)
      assert (chunk['row'] <= max_row_id)
      assert (chunk['start'] != chunk['end']) # We don't want singleton chunks
      chunks.append(chunk)
    return chunks

def _get_num_chunks(row_size, chunk_size):
    # DEPRECATED
    # See description of _get_chunk_map()
    """
    _get_chunk_map() helper 
    """
    if row_size > chunk_size:
        num_chunks = int(row_size/chunk_size)
    else:
        num_chunks = 1
    return num_chunks

def _get_chunk_map(l_info, chunk_size):
    # DEPRECATED
    # This method assumes the larger dimension to be the column. However, to
    # make changes to the model, we need to know **exactly** where the param is
    # located. When we are oblivious to whether a dim is the col or row, i.e.,
    # the 2nd or the 1st dim, we lose location information. Hence this approach
    # should not be used.
    """
    Generates the chunk map.
    """

    one_d_layer_chunks = []
    two_d_layer_chunks = []

    logger.info("(Generating Chunk Map) Processing layers...")
    for layer in tqdm(l_info.keys()):

        assert (l_info[layer]['num_dims'] <= 2)

        if l_info[layer]['num_dims']==1:

            # Use col_id to represent larger axis
            if l_info[layer]['len_dim1'] >= l_info[layer]['len_dim2']:
                num_cols = l_info[layer]['len_dim1'] # (also the size of a row)
                num_rows = 1 
            else:
                num_cols = l_info[layer]['len_dim2'] # (also the size of a row)
                num_rows = 1 

            # Process row to get chunks 
            i = 0
            j = num_cols - 1
            row_id = 0

            # Calculate num_chunks
            num_chunks = _get_num_chunks(num_cols, chunk_size)

            one_d_layer_chunks = _get_chunks(layer, row_id, i, j, max_row_id=0, num_chunks=num_chunks)

        elif l_info[layer]['num_dims']==2:

            # Use col_id to represent larger axis
            if l_info[layer]['len_dim1'] >= l_info[layer]['len_dim2']:
                num_cols = l_info[layer]['len_dim1']
                num_rows = l_info[layer]['len_dim2']
            else:
                num_cols = l_info[layer]['len_dim2']
                num_rows = l_info[layer]['len_dim1']
                
            # Process rows to get chunks 
            i = 0
            j = num_cols - 1

            # Calculate num_chunks
            num_chunks = _get_num_chunks(num_cols, chunk_size)

            for row_id in range(0, num_rows):
              two_d_layer_chunks += _get_chunks(layer, row_id, i, j, max_row_id=num_rows-1, num_chunks = num_chunks)

    all_chunks = one_d_layer_chunks + two_d_layer_chunks
    chunk_map = {index: value for index, value in enumerate(all_chunks)}
    return chunk_map
