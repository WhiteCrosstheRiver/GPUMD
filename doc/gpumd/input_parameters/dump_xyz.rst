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

* Then one can write the properties to be output, and the allowed properties include: :attr:`mass`, :attr:`velocity`, :attr:`speed`, :attr:`force`, :attr:`force_norm`, :attr:`potential`, :attr:`virial`, :attr:`group`, :attr:`unwrapped_position`, :attr:`volume`, :attr:`stress`, :attr:`stress_norm`, and :attr:`pressure`. Unknown tokens are rejected. For :attr:`stress`, :attr:`stress_norm`, and :attr:`pressure`, optional Voronoi :math:`R` and direction count must be omitted entirely or supplied as a complete, valid pair.

* The wrapped positions will always be included in the output.

* :attr:`volume` requests per-atom ball-restricted Voronoi volumes and must be followed by two numbers: the restriction radius :math:`R` (Å) and the number of quadrature directions (currently 128 or 256). Example::

    dump_xyz -1 0 100 dump.xyz volume 3.0 128

  The volume of atom :math:`i` is :math:`\Omega_i(R)=\mathrm{Vol}[C_i\cap B(\mathbf r_i,R)]`, approximated by equal-weight spherical quadrature. Neighbors are taken from a dedicated list with cutoff :math:`2R`. Each periodic box thickness must be at least :math:`4R` so that the minimum-image convention covers that neighborhood. :math:`R` should cover the farthest vertex of a bulk Voronoi cell; using half a bond length is generally too small. Volume is computed only on dump steps and does not require per-atom virial.

* :attr:`stress` writes the per-atom stress tensor (9 components, eV/Å³) as :math:`\sigma_i=(W_i+m_i\mathbf v_i\mathbf v_i^{\mathsf T})/\Omega_i(R)`, using the same virial partitioning and kinetic convention as the header ``stress=`` field. Component order matches :attr:`virial`: ``xx xy xz yx yy yz zx zy zz``. It needs the same Voronoi :math:`R` and direction count as :attr:`volume`. Give those numbers after :attr:`stress` when volume is not requested; if :attr:`volume` already supplied them, write :attr:`stress` alone. Examples::

    dump_xyz -1 0 100 dump.xyz stress 3.0 128
    dump_xyz -1 0 100 dump.xyz volume 3.0 128 stress virial

  Conflicting :math:`R` or direction counts are an error. The ``volume_atom`` column is written only when :attr:`volume` is present. Header ``stress=`` is always written, with or without this keyword.

* :attr:`speed`, :attr:`force_norm`, and :attr:`stress_norm` are independent scalar fields. They reuse the velocity, force, and per-atom stress data when those are also requested, and do not change the component counts of :attr:`velocity` (:math:`3`), :attr:`force` (:math:`3`), or :attr:`stress` (:math:`9`). :attr:`speed` is in Å/fs. :attr:`force_norm` is :math:`\sqrt{F_x^2+F_y^2+F_z^2}`. :attr:`stress_norm` is the Frobenius norm of the 9-component per-atom stress, :math:`\sqrt{\sum_{\alpha\beta}\sigma_{\alpha\beta}^2}` (eV/Å³). :attr:`stress_norm` needs the same Voronoi parameters as :attr:`stress`. Example::

    dump_xyz -1 0 100 dump.xyz velocity speed force force_norm stress 3.0 128 stress_norm

* :attr:`pressure` writes a scalar pressure in GPa (compression positive) for the whole box and for each atom. The per-atom value is :math:`p_i=(C_P/3)\,\mathrm{Tr}\,\sigma_i` with :math:`C_P` equal to ``PRESSURE_UNIT_CONVERSION``, so it matches the dumped per-atom stress (eV/Å³, compression positive) after unit conversion. The header ``pressure=`` is :math:`P_{\mathrm{box}}=C_P(\sum_i A_i)/(3V_{\mathrm{box}})`, where :math:`A_i=\mathrm{Tr}\,\mathbf W_i+m_i|\mathbf v_i|^2`. It needs the same Voronoi :math:`R` and direction count as :attr:`volume` or :attr:`stress`. Global pressure always uses every atom and the simulation-box volume, even if only a group is dumped. Examples::

    dump_xyz -1 0 100 dump.xyz pressure 3.0 128
    dump_xyz -1 0 100 dump.xyz volume 3.0 128 stress pressure


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
