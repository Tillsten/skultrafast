.. role:: bi
  :class: bolditalic


Creating publication-ready figures
==================================

While the ad-hoc settings of skultrafast are generally fine, some
additional adjustment may be necessary to create figures for an article.
Here, in this tutorial will the most common steps. If you think anything
is missing or wrong, don't hesitate to submit an issue or pull-request.
If you looking for way to do something, please also have a look at the
`matplotlib documentation <https://matplotlib.org/devdocs/index.html>`__.
It contains many examples and multiple tutorials.

\*\ **tl;dr**

Call ``plot_helpers.enable_style()`` before creating your figures.

Checklist
---------

- [ ] Figure size set according to journal guidelines
- [ ] Font selected according to journal guidelines
- [ ] (optionally) mathfont selected accordingly to the main font
- [ ] Distinguishable colors
- [ ] Vector format used
- [ ] Set rasterized ``True`` for large images and colermaps in the figure
- [ ] No part of the figure is clipped
- [ ] Consistent frequency axes: Identical units and directions

[TOC]

Figure size
-----------

**Use the journal supplied figure size.**

The most important step is to get the figure size correct. By default,
matplotlib creates a (4 in x 3 in) figure (1in = 2.54 cm) which is a reasonable
size for the screen but unsuitably large on paper. If the figure is just
manually scaled down after creation, the font-size is too small and other small
details like ticks may be not recognizable anymore.

Instead, most journals author guidelines give values for the maximum width of
figures, use them. It is not always necessary to use the full width, sometimes
its okay to leave some free space. As a rule of thump, a single column figure
should have width of around 3 in, a two column figure about 6.5".

If the figure is too small on screen, which is often the case in the
Jupyter-notebook, resist the urge to make the figure larger. Instead, increase
the dpi of the figure. The matplotlib defaults of 100 dpi are quite low compared
to modern screens. Desktop-screens often have around 144 dpi and laptop screens
can get up 230 dpi.

So far, we only discussed the width of a figure, what about the height?

Figure composition
------------------

**Try to create composite figures by code.**

Most figures in journals are multi-panel figures. There are a two ways to create
such figures: either we create the single figures first and do the composition
of the figures with a vector graphics program like *Inkscape* or we create the
the multi-panal graphic directly in matplotlib. So which one should be used?

Using a vector graphics program has several advantages: *wysiwyg*, easy
fine-tuning and mouse support. But this is bought with some serve drawbacks: if
you need change on of the sub-figures, you need to adjust the rest of the figure
as well. Also, if you have to change fonts due to a resubmission, you have to
apply it to both, the single figures and later added graphical elements. Also,
version control not commonly supported for graphics format and exactly
recreating a figure requires a lot manual steps.

Hence, if possible, do the whole figure in matplotlib. Initially, this can
result if a lot manual adjustment for things like text labels. This can be often
circumvented by using the text alignment settings and changing the
transformation.

So how to built a more complex layout ? Matplotlib offers multiple ways to
layout figures. The most flexible way is use a `gridspec
<https://matplotlib.org/3.2.1/tutorials/intermediate/gridspec.html>`__. For
simple cases, the ``plt.sublots`` function. It supports the sharing of an axis
and also takes Avoid the ``plt.subplot`` function.

Font selection
--------------

**Select a font identical or similar to the font of the publication.
When in doubt, choose Arial or Helvetica.**

The default font of matplotlib is DejaVu Sans. Its advantage it is free license
and its great coverage of Unicode glyphs. But it looks quite unique and hence
conflicts with the rest of document. Therefore I strongly advocate to change the
font. Most journals have their own preference. If the journal does not propose a
font, I suggest to use *Arial, Helvetica* or *TeX Gyre Heros*. While this may
lead to an boring figure, it also looks professional. Most journals have
guidelines for the font size in figures, use them by directly adjusting the
rcParams. For additional fine adjustment try to use relative font sizes.

Colors
------

**Make your colors distinguishable.**

The choice of colors is matter of preferences, hence different people
like different color-cycles. In general, the default matplotlib
color-cycle works quite well. Still many people prefer other colors,
e.g. the matlab color cycle, or you want to go for a more unique look.
As long as you choose easily distinguishable colors, your are fine.
Remember that figures are quite often used in presentations, hence to
avoid remaking figures don't use bright colors on a white background. In
general, the color should be chosen with contrast in mind.

The color cycle is set via the axes cycler. See the `matplotlib
documentation <https://https://matplotlib.org/3.2.1/tutorials/intermediate/color_cycle.html>`__.
Note that matploblib also supports xkcd ans CSS `color
names <https://matplotlib.org/3.2.1/tutorials/colors/colors.html>`__.

File Format
-----------

**Use vector formats. Rasterize large artists.**

Luckily, most publishers accept or require the figures in a vector format.
Therefore save the figure as an svg oder pdf file. If you somehow must supply a
pixel format, use png and make sure the dpi is set to at least 300.

In general, using a vector format also reduces the file size. If the plot
contains complex colormaps or images, it is appropriate to rasterize the
corresponding artists. That means that these are embedded as an pixel graphic
within the vector figure, thus decreasing the file size and the rendering time
enormously.


