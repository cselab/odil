#!/usr/bin/env pvbatch

# state file generated using paraview version 5.8.0

from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

import argparse
import numpy as np
import matplotlib.pyplot as plt

import paratools

parser = argparse.ArgumentParser(description="Renders 2D field")
parser.add_argument('files', nargs='+', help="List of data files 'chi_*.xmf'")
parser.add_argument('--force',
                    action="store_true",
                    help="Overwrite existing files")
parser.add_argument('--res',
                    type=int,
                    nargs=2,
                    default=[512, 256],
                    help="Output image resolution")
parser.add_argument('--range',
                    type=float,
                    nargs=2,
                    default=[0, 1],
                    help="Range of values")
parser.add_argument('--cmap', type=str, default='viridis', help="Colormap")
parser.add_argument('--name',
                    type=str,
                    default='chi',
                    help="Output filename prefix and field name")
parser.add_argument('--omega',
                    action="store_true",
                    help="Render vorticity from "
                    "velocity vx_*.xmf and vy_*.xmf")
parser.add_argument('--cells', type=int, default=1)
parser.add_argument('--discrete',
                    type=int,
                    default=0,
                    help="Number of discrete values")
parser.add_argument('--imposed',
                    type=str,
                    default=None,
                    help="Path to imposed.csv")
args = parser.parse_args()

sources_ft = []
timearrays = []

files_chi = args.files
source_chi = XDMFReader(FileNames=files_chi)
(source_chi, ), (timearray, ) = paratools.ApplyForceTime([source_chi])
sources_ft.append(source_chi)
timearrays.append(timearray)

outfield = 'out'

if args.omega:
    files_vx = paratools.ReplaceFilename(args.files, "vx_{}.xmf")
    files_vy = paratools.ReplaceFilename(args.files, "vy_{}.xmf")
    sources_vxy = (XDMFReader(FileNames=files_vx),
                   XDMFReader(FileNames=files_vy))
    sources_vxy, timearrays_vxy = paratools.ApplyForceTime(sources_vxy)
    sources_ft += sources_vxy
    timearrays += timearrays_vxy
    appendAttributes1 = AppendAttributes(Input=sources_vxy)
    calculator1 = Calculator(Input=appendAttributes1)
    calculator1.ResultArrayName = 'vel'
    calculator1.Function = 'vx*iHat+vy*jHat'
    gradient1 = Gradient(Input=calculator1)
    gradient1.ScalarArray = ['CELLS', 'vel']
    gradient1.ComputeGradient = 0
    gradient1.ComputeVorticity = 1
    gradient1.VorticityArrayName = 'omega'
    calculator_out = Calculator(Input=gradient1)
    calculator_out.ResultArrayName = outfield
    calculator_out.Function = 'omega_Z'
else:
    calculator_out = Calculator(Input=source_chi)
    calculator_out.ResultArrayName = outfield
    calculator_out.Function = args.name

if not args.cells:
    calculator_out = CellDatatoPointData(Input=calculator_out)
    calculator_out.CellDataArraytoprocess = [outfield]

renderView1 = CreateView('RenderView')
renderView1.OrientationAxesVisibility = 0
renderView1.UseLight = 0
renderView1.ViewSize = args.res
renderView1.CameraPosition = [1, 0.5, 10]
renderView1.CameraFocalPoint = [1, 0.5, 0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.5
renderView1.CameraParallelProjection = 1
renderView1.Background = [1] * 3


def cmap_to_vrgb(name, values):
    "Returns rows of [value,r,g,b]"
    values = np.array(values)
    lin = np.linspace(0, 1, len(values))
    rgba = plt.get_cmap(name)(lin)
    res = np.hstack((values[:, None], rgba[:, :3]))
    return res


chiLUT = GetColorTransferFunction('out')

vrgb = cmap_to_vrgb(args.cmap, np.linspace(*args.range))
chiLUT.RGBPoints = vrgb.flatten()
chiLUT.ColorSpace = 'RGB'
chiLUT.ScalarRangeInitialized = 1.0
if args.discrete:
    chiLUT.Discretize = 1
    chiLUT.NumberOfTableValues = args.discrete

chiDisplay = Show(calculator_out, renderView1)
chiDisplay.Representation = 'Surface'
chiDisplay.ColorArrayName = ['CELLS' if args.cells else 'POINTS', 'out']
chiDisplay.LookupTable = chiLUT

if args.imposed:
    imposedcsv = CSVReader(FileName=[args.imposed])
    imposedPoints = TableToPoints(Input=imposedcsv)
    imposedPoints.XColumn = 'x'
    imposedPoints.YColumn = 'y'
    imposedPoints.ZColumn = 'x'
    imposedPoints.a2DPoints = 1
    imposedDisplay = Show(imposedPoints, renderView1)
    imposedDisplay.Position = [0.0, 0.0, 1.0]
    imposedDisplay.AmbientColor = [1.0, 0.121, 0.356]
    imposedDisplay.ColorArrayName = [None, '']
    imposedDisplay.DiffuseColor = [1.0, 0.121, 0.356]
    imposedDisplay.Representation = 'Points'
    imposedDisplay.PointSize = 3.0
    imposedDisplay.RenderPointsAsSpheres = 1
    '''
    imposedDisplay.Representation = 'Point Gaussian'
    imposedDisplay.GaussianRadius = 0.005
    imposedDisplay.ShaderPreset = 'Plain circle'
    '''

steps = paratools.GetSteps(args.files)
paratools.SaveAnimation(steps,
                        renderView1,
                        sources_ft,
                        timearrays,
                        pattern=args.name + '_{}.png',
                        force=args.force)
