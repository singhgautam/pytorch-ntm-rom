# Copy Task with Neural Turing Machine having Read-Only Memory

Neural Turing Machine (NTM) has a controller that 'controls' the read and write operation on an external memory. As a beginner, it is complicated to have memory acted upon by both read and write heads. Instead, this repository implements NTM with controller controlling only the read operation. The write operation is hard-wired and every new input seen by the NTM is simply appended to the next row in the memory.
