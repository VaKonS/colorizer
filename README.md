# colorizer
Implements different methods to transfer colors between images.

Continuation of work on [color transferring function](https://github.com/VaKonS/neural-style/blob/f36f8fc3db999ab3612bc03fd80032a5e15584b1/neural_style.lua#L524-L801) for "[neural-style](https://github.com/jcjohnson/neural-style)". Since it can work alone, maybe someone will find it useful. 

**Requirements:**
- Torch 7 (https://github.com/torch/torch7);
- Torch Image (https://github.com/torch/image).

**License:**

Public domain.

**Usage:**

`th colorizer.lua -palette_image "gradients.png" -colorized_image "sausage.jpg" -output_image "yummy.png" -recolor_strength 1 -color_function "lab"`

- `-palette_image` – image to take colors from;
- `-colorized_image` – image to recolor;
- `-output_image` – image to save result;
- `-recolor_strength` – -N ... 0 ... 1 ... N, new palette strength, from original colors (0) to fully recolored (1);
- `-color_function` – color transfer mode, currently present are: chol, pca, sym/mkl, rgb, xyz, lab, lms, hsl, hsl-full, hsl-polar, hsl-polar-full, lab-rgb, chol-pca, chol-sym, exp1.

**Experimental**, don't expect miracles from it.

Based on [Leon Gatys's](https://github.com/leongatys/NeuralImageSynthesis), [ProGamerGov's](https://github.com/jcjohnson/neural-style/issues/376), [Adrian Rosebrock's](https://github.com/jrosebr1/color_transfer), [François Pitié's](https://github.com/frcs/colour-transfer), [mdfirman's](https://github.com/mdfirman/python_colour_transfer) code and different research papers (given in descriptions of functions).

**Example**:

![Example of color transfer](https://github.com/VaKonS/colorizer/blob/master/sausage_cpsmccrxllhhhhle.jpg)
