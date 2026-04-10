Take the `trace_function_to_cuda` module and expand its functionality so that instead of only processing one input at a time, it processes a whole folders worth of input.

Currently, `trace_function_to_cuda` takes an input python file, an input json, and a path to produce the output json.
I want you to modify it so that the core functionality stays the same, but it instead takes a path to a folder of python files, a path to a folder of input json, and a path to a folder to write output json to.

Specifically, I want to be able to invoke the file as such:
```
/KernelBench/.venv/bin/python -m trace_functional_to_cuda output_collated kcheck_specs kcheck_tests
```

Where output_collated has this structure
output_collated
- level 1
  - <problem_id>
    - correct
      - <solution_id>
      - ...
    - incorrect
      - <solution_id>
    - ...
 - ...
- level 2
  - ...same as level 1

And kcheck_specs has this structure
kcheck_specs
- level1
 - <problem_name>.json
- level2
 - <problem_name>.json

And kcheck_tests should be created and mirror the structure of output_collated, with each json file named with ascending ids.

The core logic should be run once for each solution file in output_collated. Note that there is only one input json per problem_id, so they will have to be broadcast to the many solutions that exist for a given problem.

In addition to outputting the json file to `kcheck_specs`, I would also like you to copy over the `<problem_name>.cpp` file from kcheck_specs once per problem.

You can keep the core logic exactly the same as before, but add a layer on top that resolves file paths and passes them to the main functionality.

By default, the script should skip processing an input if the destination file is already there. This can be skipped with the `--force` flag.

In addition, add multiprocessing, to process tasks quicker, and a progress bar, so that we know how far through the script is.