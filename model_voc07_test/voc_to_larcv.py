import ROOT
import numpy
from ROOT import larcv
larcv.load_pyutil()
from voc_data_loader import voc_meta, voc_image


def main():
    # # Set up larcv saver:

    io = larcv.IOManager(larcv.IOManager.kWRITE)
    io.set_out_file('voc_larcv_test.root')
    io.initialize()

    _top_level = "VOC2007"
    _base_name = "voc07_larcv"

    # Now, need to load the images
    _voc_metadata = voc_meta("VOC2007")

    # Get the first image:
    train_all_indexes = _voc_metadata.train_indexes(None, _seg_indexes=False)
    train_seg_indexes = _voc_metadata.train_indexes(None, _seg_indexes=True)
    print len(indexes)
    for index in indexes[4:10]:
        # image = indexes[index]
        _xml =  "{:06d}.xml".format(index)
        img = voc_image('VOC2007', _xml)
        # image_to_larcv(img, io, run=7)

    val_all_indexes = _voc_metadata.val_indexes(None, _seg_indexes=False)
    val_seg_indexes = _voc_metadata.val_indexes(None, _seg_indexes=True)
    print len(indexes)

    io.finalize()




def image_to_larcv(img, io, run=1, include_segmentation=False):

    image2d_array = io.get_data('image2d', 'voc')
    particle_array = io.get_data('particle', 'voc')
    if include_segmentation:
        object_array = io.get_data('cluster2d', 'voc_obj')
        class_array = io.get_data('cluster2d', 'voc_cls')

    # Image2d creation:
    # _img = numpy.rot90(img.image(), 3)
    _img = numpy.transpose(img.image(), axes=(1,0,2))
    n_x = _img.shape[0]
    n_y = _img.shape[1]
    for i in xrange(3):
        meta = larcv.ImageMeta(0,0,
                               n_x, n_y,
                               n_y, n_x,
                               i)
        img_i = _img[:,:,i]
        img_i = numpy.fliplr(img_i)
        out_vec = ROOT.std.vector("float")()
        out_vec.reserve(n_x*n_y)
        for x in range(n_x):
            for y in range(n_y):
                out_vec.push_back(img_i[x,y])
        image2d_array.append( larcv.Image2D(meta, out_vec))

    # "Particle" creation:
    for box, category in zip(img.bounding_boxes(), img.categories()):
        _part = larcv.Particle()
        _part.pdg_code(_voc_metadata.class_index(category))
        for i in xrange(3):
            _bbox = larcv.BBox2D(box[0], n_y - box[3], box[2], n_y - box[1], i)
            # _bbox = larcv.BBox2D(box[0], box[1], box[2], box[3], i)
            _part.boundingbox_2d(_bbox, i)
        # print type(category)
        _part.creation_process(str(category))
        particle_array.append(_part)

    # Segmentation creation:
    if include_segmentation:
        if img.has_segmentation():
            #Segmentation objects:

            _seg_obj_image = numpy.fliplr(numpy.transpose(img.segmentation_object(), axes=(1,0)))
            _seg_cls_image = numpy.fliplr(numpy.transpose(img.segmentation_class(), axes=(1,0)))

            meta = larcv.ImageMeta(0,0,
                                   n_x, n_y,
                                   n_y, n_x,
                                   0)
            # Object segmentation:
            labels = numpy.unique(_seg_obj_image)
            clusterpixel2d_obj = larcv.ClusterPixel2D()
            clusterpixel2d_obj.meta(meta)
            print labels
            for label in labels:
                if label == 0:
                    continue
                voxset = larcv.VoxelSet()

                label_x, label_y = numpy.where(_seg_obj_image == label)
                print label_x.shape
                for _x, _y in zip(label_x, label_y):
                    voxset.add(larcv.Voxel(meta.index(_y,_x), 1))


                clusterpixel2d_obj.as_vector().push_back(voxset)
                voxset.id(int(label))

            object_array.set(clusterpixel2d_obj)

            # Class segmentation:
            labels = numpy.unique(_seg_cls_image)

            clusterpixel2d_cls = larcv.ClusterPixel2D()
            clusterpixel2d_cls.meta(meta)

            print labels
            for label in labels:
                if label == 0:
                    continue
                voxset = larcv.VoxelSet()
                label_x, label_y = numpy.where(_seg_cls_image == label)
                for _x, _y in zip(label_x, label_y):
                    voxset.add(larcv.Voxel(meta.index(_y,_x), 1))

                voxset.id(int(label))
                clusterpixel2d_cls.as_vector().push_back(voxset)

            class_array.set(clusterpixel2d_cls)

    io.set_id(run,0,index)
    io.save_entry()




if __name__ == '__main__':
    main()