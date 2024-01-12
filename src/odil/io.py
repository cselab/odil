import os
import re
import numpy as np


def parse_raw_xmf(xmfpath):
    '''
    Returns shape and path to `.raw` file
    xmfpath: path to `.xmf` metadata file
    '''
    with open(xmfpath) as fin:
        text = ''.join(fin.read().split('\n'))
    m = re.findall(
        '<Xdmf.*<Attribute.*'
        '<DataItem.*<DataItem.*'
        '<DataItem.*Dimensions="(\d*) (\d*) (\d*)".*?> *([a-z0-9_.]*)',
        text)[0]
    shape = tuple(map(int, m[:3]))
    rawpath = m[3]
    rawpath = os.path.join(os.path.dirname(xmfpath), rawpath)
    return shape, rawpath


def read_raw(xmfpath):
    '''
    Returns array from scalar field in raw format.
    xmfpath: path to xmf metadata file
    '''
    shape, rawpath = parse_raw_xmf(xmfpath)
    u = np.fromfile(rawpath).reshape(shape)
    return u


def write_raw_xmf(xmfpath,
                  rawpath,
                  count,
                  spacing=(1, 1, 1),
                  name='data',
                  precision=8,
                  cell=True):
    '''
    Writes XMF metadata for a `.raw` datafile.
    xmfpath: path to output `.xmf` file
    rawpath: path to binary `.raw` file to be linked
    count: array size as (Nz, Ny, Nx)
    name: name of field
    cell: cell-centered field, else node-centered
    '''

    txt = '''\
<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
 <Domain>
   <Grid Name="mesh" GridType="Uniform">
     <Topology TopologyType="{dim}DCORECTMesh" Dimensions="{nodes*}"/>
     <Geometry GeometryType="{geomtype}">
       <DataItem Name="Origin" Dimensions="{dim}" NumberType="Float" Precision="8" Format="XML">
         {origin*}
       </DataItem>
       <DataItem Name="Spacing" Dimensions="{dim}" NumberType="Float" Precision="8" Format="XML">
         {spacing*}
       </DataItem>
     </Geometry>
     <Attribute Name="{name}" AttributeType="Scalar" Center="{center}">
       <DataItem ItemType="HyperSlab" Dimensions="{countd*}" Type="HyperSlab">
           <DataItem Dimensions="3 {dim}" Format="XML">
             {start*}
             {stride*}
             {count*}
           </DataItem>
           <DataItem Dimensions="{bindim*}" Seek="{seek}" Precision="{precision}" NumberType="{type}" Format="Binary">
             {binpath}
           </DataItem>
       </DataItem>
     </Attribute>
   </Grid>
 </Domain>
</Xdmf>
'''

    def tostrrev(v):
        return ' '.join(map(str, reversed(v)))

    def tostr(v):
        return ' '.join(map(str, v))

    info = dict()
    dim = 3
    info['name'] = name
    info['dim'] = dim
    info['origin'] = tostrrev([0] * dim)
    info['spacing'] = tostrrev(spacing)
    info['start'] = tostrrev([0] * dim)
    info['stride'] = tostrrev([1] * dim)
    info['count'] = tostr(count)
    info['bindim'] = tostr(count)
    info['countd'] = tostr(count)
    if cell:
        info['nodes'] = tostr([a + 1 for a in count])
        info['center'] = 'Cell'
    else:
        info['nodes'] = tostr([a for a in count])
        info['center'] = 'Node'
    info['precision'] = precision
    if precision == 8:
        info['type'] = 'Double'
    else:
        info['type'] = 'Float'
    info['binpath'] = rawpath
    info['seek'] = '0'
    info['geomtype'] = 'ORIGIN_DXDYDZ'
    # Remove '*' which are only used in `aphros/src/dump/xmf.ipp`.
    txt = txt.replace('*}', '}')
    txt = txt.format(**info)

    with open(xmfpath, 'w') as fout:
        fout.write(txt)


def write_raw_with_xmf(u,
                       xmfpath,
                       rawpath=None,
                       spacing=(1, 1, 1),
                       cell=True,
                       name='data'):
    '''
    Writes binary data in raw format with XMF metadata.
    u: np.ndarray to write, shape (Nz, Ny, Nx)
    spacing: cell size
    name: name of field
    '''
    if len(u.shape) != 3:
        u = u.reshape((1, ) + u.shape)
    if len(spacing) != 3:
        spacing = list(spacing) + [min(spacing)]
    precision = 4 if u.dtype == np.float32 else 8
    if rawpath is None:
        rawpath = os.path.splitext(xmfpath)[0] + ".raw"
    rawrelpath = os.path.relpath(rawpath, start=os.path.dirname(xmfpath))
    write_raw_xmf(xmfpath, rawrelpath, u.shape, spacing, name, precision, cell)
    u.tofile(rawpath)
    return xmfpath


def write_vtk_poly(fout,
                   points,
                   polygons,
                   point_fields=None,
                   cell_fields=None,
                   comment="",
                   fmt='%.16g'):
    """
    Writes polygons to ASCII legacy VTK file.
    fout: `str` or file-like
        Path to output legacy VTK file or file-like object.
    points: `numpy.ndarray`, shape (npoints, 3)
        3D points.
    polygons: `list` [`list` [ `int` ]], shape (ncells, ...)
        Polygons as lists of indices in `points`.
    point_fields: `dict` from `str` to array of shape (npoints,)
        Mapping from names to arrays storing scalar fields on points.
    cell_fields: `dict` from `str` to array of shape (ncells,)
        Mapping from names to arrays storing scalar fields on cells.
    """
    path = None
    if type(fout) is str:
        path = fout
        fout = open(path, 'wb')

    def write(data):
        if type(data) is str:
            data = data.encode()
        fout.write(data)

    write("# vtk DataFile Version 2.0\n")

    write(comment + '\n')

    write("ASCII\n")

    write("DATASET POLYDATA\n")

    # Write points.
    npoints = len(points)
    write("POINTS {:} float\n".format(npoints))
    np.savetxt(fout, points, fmt=fmt)

    # Write cells.
    ncells = len(polygons)
    num_poly_data = len(polygons) + sum([len(p) for p in polygons])
    write("POLYGONS {:} {:}\n".format(ncells, num_poly_data))
    for p in polygons:
        write(' '.join(map(str, [len(p)] + p)) + '\n')

    # Write point fields.
    if point_fields is None:
        point_fields = dict()
    if point_fields:
        write("\nPOINT_DATA {:}\n".format(npoints))
    for name, array in point_fields.items():
        array = np.reshape(array, -1)
        if array.size != npoints:
            raise RuntimeError(
                f"Expected equal array.size={array.size} and npoints={npoints}"
            )
        write("SCALARS {:} float\n".format(name))
        write("LOOKUP_TABLE default\n")
        np.savetxt(fout, array, fmt=fmt)

    # Write cell fields.
    if cell_fields is None:
        cell_fields = dict()
    if cell_fields:
        write("\nCELL_DATA {:}\n".format(ncells))
    for name, array in cell_fields.items():
        array = np.reshape(array, -1)
        if array.size != ncells:
            raise RuntimeError(
                f"Expected equal array.size={array.size} and ncells={ncells}")
        write("SCALARS {:} float\n".format(name))
        write("LOOKUP_TABLE default\n")
        np.savetxt(fout, array, fmt=fmt)

    if path:
        fout.close()
