using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;

namespace train_classification_model
{
    public class DataPreProcessing
    {
        public Dictionary<string, string[]> load_dataset_files()
        {
            var animals = new Dictionary<string, string[]>();
            foreach (var animal in new[] { "dog", "cat", "elephant", "cow" })
            {
                var files = Directory.GetFiles($"data/images/{animal}");
                animals.Add(animal, files);
            }
            return animals;
        }

        public NDArray image_file_to_tensor(string file)
        {
            using (var image = Image.Load<Rgba32>(file))
            {
                var width = image.Width;
                var height = image.Height;
                int left, top, right, bottom;

                if (width > height)
                {
                    left = (width - height) / 2;
                    right = (width + height) / 2;
                    top = 0;
                    bottom = height;
                }
                else
                {
                    left = 0;
                    right = width;
                    top = (height - width) / 2;
                    bottom = (height + width) / 2;
                }

                image.Mutate(x => x.Crop(new Rectangle(left, top, right - left, bottom - top))
                            .Resize(224, 224));
                var rgbValues = new byte[224 * 224 * 3];
                for (int y = 0; y < image.Height; ++y)
                {
                    for (int x = 0; x < image.Width; ++x)
                    {
                        var pixel = image[x, y];
                        rgbValues[y * image.Width * 3 + x * 3] = pixel.R;
                        rgbValues[y * image.Width * 3 + x * 3 + 1] = pixel.G;
                        rgbValues[y * image.Width * 3 + x * 3 + 2] = pixel.B;
                    }
                }

                var pix = np.transpose(np.reshape(np.array(rgbValues, np.float32), new Shape(224, 224, 3)), [2, 0, 1]);
                pix = pix / 255.0;
                pix[0] = (pix[0] - 0.485) / 0.229;
                pix[1] = (pix[1] - 0.456) / 0.224;
                pix[2] = (pix[2] - 0.406) / 0.225;
                return pix;
            }
        }

        public Tensor<T> np_to_tensor<T>(NDArray array) where T : unmanaged
        {
            // Get the shape of the NDArray
            var shape = array.shape;
            // Get the data of the NDArray
            var data = array.ToArray<T>();
            // Create a DenseTensor from the data and shape
            return new DenseTensor<T>(data, shape);
        }

        public NDArray tensor_to_ndArray<T>(Tensor<T> tensor) where T : unmanaged
        {
            // Convert Tensor to NDArray
            var dim = tensor.Dimensions;
            var shape = new Shape(dim.ToArray());
            var ndArray = new NDArray(typeof(T), shape);
            var data = tensor.ToArray();
            ndArray.SetData(data);
            return ndArray;
        }

        public NDArray softmax2D(NDArray array2D)
        {
            var list2D = new List<NDArray>();
            for (int i = 0; i < array2D.shape[0]; i++)
            {
                list2D.Add(array2D[i]);
            }

            var list1D = new List<NDArray>();
            foreach (var array1D in list2D)
            {
                list1D.Add(softmax1D(array1D));
            }

            return np.stack(list1D.ToArray());
        }

        public NDArray softmax1D(NDArray array1D)
        {
            return np.exp(array1D) / np.exp(array1D).sum();
        }
    }
}