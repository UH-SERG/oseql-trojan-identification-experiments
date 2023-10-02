#approach = "sequential_line_chunks" # OPTIONS:- "sequential_line_chunks", "sequential_char_chunks", "ddmin_lines"
approach = "sequential_line_chunks"
chunk_size = "N/A" # not needed for ddmin_lines/sequential_line_chunks
triggers = [
        'int capacity = 5333;',
        'assert(15>=0);',
        'assert(-15<=0);',
        'int *panel_id;',
        'int zoom_ratio;'
      ]

