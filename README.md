# Copy Task with Read-Only Memory Neural Turing Machine

Neural Turing Machine (NTM) has a controller that 'controls' the read and write operation on an external memory. As a beginner,
it is complicated to work with both read and write heads. Instead, this repository implements NTM with controller
controlling only the read operation. The write operation is hard-wired and every new input seen by the NTM is simply appended
to the next row in the memory.
