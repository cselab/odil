import os
import re
import numpy as np


def parse_raw_xmf(xmfpath):
    '''
    Returns shape and path to `.raw` file
    xmfpath: path to `.xmf` metadata file
    '''
    with open(xmfpath) as f:
        text = ''.join(f.read().split('\n'))
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

    with open(xmfpath, 'w') as f:
        f.write(txt)


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
    write_raw_xmf(xmfpath, rawpath, u.shape, spacing, name, precision, cell)
    u.tofile(rawpath)
    return xmfpath
