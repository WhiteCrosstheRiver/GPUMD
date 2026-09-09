.. _kw_dump_xyz:
.. index::
   single: dump_xyz (keyword in run.in)

:attr:`dump_xyz`
================

Write per-atom data into user-specified file(s) in `extended XYZ format <https://github.com/libAtoms/extxyz>`_.

Syntax
------

.. code::

   dump_xyz <grouping_method> <group_id> <inverval> <filename> {<property_1> <property_2> ...}

* :attr:`grouping_method` and :attr:`group_id` are the grouping method and the related group ID to be used.

If :attr:`grouping_method` is negative, :attr:`group_id` will be ignored and data for the whole system will be output.

* :attr:`interval` is the output interval (number of steps) of the data.

* :attr:`filename` is the output file.

If it is ended by a star (*), the data for one frame will be output to one file, named by changing the star to the step number.

* Then one can write the properties to be output, and the allowed properties include: :attr:`mass`, :attr:`velocity`, :attr:`force`, :attr:`potential`, :attr:`virial`, :attr:`group`, :attr:`unwrapped_position`, and :attr:`volume`.

* The wrapped positions will always be included in the output.

* :attr:`volume` requests per-atom ball-restricted Voronoi volumes and must be followed by two numbers: the restriction radius :math:`R` (Å) and the number of quadrature directions (currently 128 or 256). Example::

    dump_xyz -1 0 100 dump.xyz volume 3.0 128

  The volume of atom :math:`i` is :math:`\Omega_i(R)=\mathrm{Vol}[C_i\cap B(\mathbf r_i,R)]`, approximated by equal-weight spherical quadrature. Neighbors are taken from a dedicated list with cutoff :math:`2R`. Each periodic box thickness must be at least :math:`4R` so that the minimum-image convention covers that neighborhood. :math:`R` should cover the farthest vertex of a bulk Voronoi cell; using half a bond length is generally too small. Volume is computed only on dump steps and does not require per-atom virial.


Examples
--------

.. code::

    ensemble xxx # some ensemble

    # dump positions every 1000 steps, for the whole system:
    dump_xyz -1 1 1000 positions.xyz

    # dump many other quantities every 100 steps, for atoms in group 0 of grouping method 1:
    dump_xyz 1 0 100 properties.xyz mass velocity potential force virial    

    run 1000000

Caveats
-------
* This keyword is not propagating.
  That means, its effect will not be passed from one run to the next.
* The output file has an appending behavior.
* Different from many of the other keywords, this keyword is allowed to be invoked multiple times within one run.
