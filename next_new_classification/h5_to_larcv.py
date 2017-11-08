import ROOT
import numpy
import h5py
import tables
from ROOT import larcv
import glob
import os, sys

def main():
    # # Set up larcv saver:

    top_path = "/data/NEXT/NEW/1M_v0_08_07/"
    out_dir = "/data/NEXT/NEW/1M_v0_08_07_larcv2/"

    _files = glob.glob(top_path + "/*")

    for _file in _files:
        print "Working on file {}".format(_file)
        #Create the output root filename:
        _base_file = os.path.basename(_file)
        _name_root = os.path.splitext(_base_file)[0]
        _out_file = out_dir + _name_root + ".root"
        # Get the run which is the last digits of the root name:
        run = int(_name_root.split("_")[-1])

        # Determine if this is signal or background:
        if 'bg' in _name_root:
            subrun = 0
        else:
            subrun = 1

        # Create an instance of io manager:
        io = larcv.IOManager(larcv.IOManager.kWRITE)
        io.set_out_file(_out_file)

        io.initialize()

        # Create the output data product holders:



        _meta = larcv.Voxel3DMeta()
        _meta.set(0,   0,  0,
                  20, 20, 60,
                  20, 20, 60)


        # Open the file and get the things out that we need:
        _f = tables.open_file(_file, 'r')
        for i in xrange(len(_f.root.energies)):
            voxel3d_array = io.get_data('sparse3d', 'next')
            particle_array = io.get_data('particle', 'next')

            voxel3d_array.meta(_meta)
            _map = _f.root.maps[i]

            x, y, z = numpy.where(_map != 0)
            vals = _map[x,y,z]
            # Loop over the values and add them to the sparse array:
            for j in xrange(len(vals)):
                voxel3d_array.emplace(x[j],y[j],z[j],vals[j])
            # "Particle" creation:
            _part = larcv.Particle()
            _part.pdg_code(subrun)
            if subrun == 0:
                _part.creation_process("background")
            else:
                _part.creation_process("signal")

            particle_array.append(_part)


            io.set_id(run,subrun,i)
            io.save_entry()
        _f.close()

        io.finalize()

    



if __name__ == '__main__':
    main()