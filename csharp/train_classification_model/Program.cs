using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using System.Runtime.InteropServices;

namespace train_classification_model
{
    class Program
    {
        static void Main(string[] args)
        {
            var dataPreProcessing = new DataPreProcessing();

            var animals = dataPreProcessing.load_dataset_files();
            var label_to_id_map = new Dictionary<string, Int64>
            {
                { "dog", 0 },
                { "cat", 1 },
                { "elephant", 2 },
                { "cow", 3 }
            };

            Console.WriteLine($"Training the model using training_artifacts/training_model.onnx...");
            //using (var gpuSessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(0))
            using (var checkpoint_state = CheckpointState.LoadCheckpoint("training_artifacts/checkpoint"))
            //using (var model = new TrainingSession(gpuSessionOptions,
            //                                checkpoint_state,
            //                                "training_artifacts/training_model.onnx",
            //                                "training_artifacts/eval_model.onnx",
            //                                "training_artifacts/optimizer_model.onnx"))
            using (var model = new TrainingSession(checkpoint_state,
                                                   "training_artifacts/training_model.onnx",
                                                   "training_artifacts/eval_model.onnx",
                                                   "training_artifacts/optimizer_model.onnx"))
            {
                //Training Loop
                var num_samples_per_class = 20;
                var num_epochs = 5;
                for (var epoch = 0; epoch < num_epochs; ++epoch)
                {
                    var loss = 0.0f;
                    for (var index = 0; index < num_samples_per_class; ++index)
                    {
                        var batch = new List<NDArray>();
                        var labels = new List<NDArray>();
                        foreach (var animal in animals)
                        {
                            batch.Add(dataPreProcessing.image_file_to_tensor(animal.Value[index]));
                            labels.Add(label_to_id_map.GetValueOrDefault(animal.Key));
                        }

                        var b = np.stack(batch.ToArray());
                        var l = np.squeeze(np.stack(labels.ToArray()));

                        var bTensor = dataPreProcessing.np_to_tensor<float>(b);
                        var lTensor = dataPreProcessing.np_to_tensor<Int64>(l);

                        using (var bValue = FixedBufferOnnxValue.CreateFromTensor(bTensor))
                        using (var lValue = FixedBufferOnnxValue.CreateFromTensor(lTensor))
                        {
                            var inputs = new List<FixedBufferOnnxValue> {
                                bValue,
                                lValue
                            };
                            using (var outputs = model.TrainStep(inputs))
                            {
                                model.OptimizerStep();
                                model.LazyResetGrad();
                                loss += outputs.ElementAt(0).AsTensor<float>().GetValue(0);
                            }
                        }
                    }
                    Console.WriteLine($"Epoch {epoch + 1} Loss {loss / num_samples_per_class}");
                }

                //Export model
                var outputNames = new List<string> { "output" };
                var exportModelPath = "inference_artifacts/trained_model.onnx";
                try
                {
                    Console.WriteLine($"Export model to {exportModelPath}...");
                    var dir = Path.GetDirectoryName(exportModelPath);
                    if (!string.IsNullOrEmpty(dir))
                    {
                        Directory.CreateDirectory(dir);
                    }
                    model.ExportModelForInferencing(exportModelPath, outputNames);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex);
                    return;
                }
            }

            //Inference
            Console.WriteLine("Inference using the trained model...");
            var openImages = new string[] { "data/images/test/test1.jpg" };

            using (var model = new InferenceSession("inference_artifacts/trained_model.onnx"))
            //using (var model = new InferenceSession("training_artifacts/mobilenetv2.onnx"))
            {
                var tensorList = new List<NDArray>();
                foreach (var file in openImages)
                {
                    tensorList.Add(dataPreProcessing.image_file_to_tensor(file));
                }
                var tensor = dataPreProcessing.np_to_tensor<float>(np.stack(tensorList.ToArray()));
                var inputName = model.InputMetadata.Keys.FirstOrDefault();
                if (!string.IsNullOrEmpty(inputName))
                {
                    var value = NamedOnnxValue.CreateFromTensor(inputName, tensor);
                    var inputs = new List<NamedOnnxValue> { value };
                    using (var outputs = model.Run(inputs))
                    {
                        var t = outputs.First().AsTensor<float>();
                        var array = dataPreProcessing.tensor_to_ndArray(t);
                        var prediction = dataPreProcessing.softmax2D(array) * 100;
                        Console.WriteLine($"test\tdog\tcat\telephant\tcow");
                        foreach (var file in openImages.Select((f, i) => (f, i)))
                        {
                            Console.WriteLine($"{Path.GetFileName(file.f)}\t" +
                                             $"{prediction.GetValue<float>(new int[] { file.i, 0 }):F2}\t" +
                                             $"{prediction.GetValue<float>(new int[] { file.i, 1 }):F2}\t" +
                                             $"{prediction.GetValue<float>(new int[] { file.i, 2 }):F2}\t" +
                                             $"{prediction.GetValue<float>(new int[] { file.i, 3 }):F2}\t");
                        }
                    }
                }
            }
        }    
    }
}