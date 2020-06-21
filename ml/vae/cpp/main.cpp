#include <torch/torch.h>
#include <ATen/ATen.h>

#include <tuple>
#include <cmath>
#include <iostream>

static const std::string MNIST_PATH = "{ABSOLUTE_PATH_TO_MNIST}";
static const int EPOCHS = 10;
static const int BATCH_SIZE= 64;
static const int LOG_INTERVAL = 10;

class Net : public torch::nn::Module
{
 public:
  Net()
    : fc1(784, 400)
    , fc21(400, 20)
    , fc22(400, 20)
    , fc3(20, 400)
    , fc4(400,784)
  {
    register_module("fc1", fc1);
    register_module("fc21", fc21);
    register_module("fc22", fc22);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
    auto h1 = torch::relu(fc1(x.view({-1, 784})));
    auto mu = fc21(h1);
    auto logvar = fc22(h1);
    auto std = at::exp(0.5*logvar);
    auto eps = at::randn_like(std);
    auto z = mu + eps * std;
    auto h3 = torch::relu(fc3(z));

    return std::make_tuple(at::sigmoid(fc4(h3)), mu, logvar);
  }

 private:
  torch::nn::Linear fc1;
  torch::nn::Linear fc21;
  torch::nn::Linear fc22;
  torch::nn::Linear fc3;
  torch::nn::Linear fc4;
};

template <typename DataLoader>
void train(
  int32_t epoch,
  torch::Device device,
  Net& model,
  DataLoader& data_loader,
  torch::optim::Adam& optimizer)
{
  model.train();

  size_t batch_index = 0;
  size_t train_loss = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device);
    optimizer.zero_grad();

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ret = model.forward(data);
    torch::Tensor recon_batch = std::get<0>(ret);
    torch::Tensor mu = std::get<1>(ret);
    torch::Tensor logvar = std::get<2>(ret);

    // loss_function
    auto BCE = at::binary_cross_entropy(recon_batch, data.view({-1, 784}), {}, at::Reduction::Sum);
    auto KLD = -0.5 * at::sum(1 + logvar - mu.pow(2) - logvar.exp());
    auto loss = BCE + KLD;
    loss.backward();
    train_loss += loss.itemsize();

    optimizer.step();
    if (batch_index++ % LOG_INTERVAL == 0) {
      std::string save_file = "model.pt";
      torch::serialize::OutputArchive output_archive;
      model.save(output_archive);
      output_archive.save_to(save_file);
    }
  }
}

template <typename DataLoader>
void test(
  int32_t epoch,
  torch::Device device,
  Net& model,
  DataLoader& data_loader)
{
  torch::NoGradGuard no_grad;
  model.eval();

  size_t batch_index = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device);
    auto targets = batch.target.to(device);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ret = model.forward(data);
    torch::Tensor recon_batch = std::get<0>(ret);
    torch::Tensor mu = std::get<1>(ret);
    torch::Tensor logvar = std::get<2>(ret);

    // loss_function
    auto BCE = at::binary_cross_entropy(recon_batch, data.view({-1, 784}), {}, at::Reduction::Sum);
    auto KLD = -0.5 * at::sum(1 + logvar - mu.pow(2) - logvar.exp());
    auto loss = BCE + KLD;

    if (batch_index == 0) {
      auto n = std::min(data.size(0), (int64_t)8);

      std::vector<torch::Tensor> datas;
      for (int64_t i = 0; i < n; ++i) {
        datas.push_back(data[i]);
      }
      auto recon_data = recon_batch.view({BATCH_SIZE, 1, 28, 28});
      for (int64_t i = 0; i < n; ++i) {
        datas.push_back(recon_data[i]);
      }

      torch::TensorList tensor_list(datas);
      auto comparison = at::cat(tensor_list);
      auto image = comparison.cuda();
      image.clone().clamp(0, 255).nonzero_numpy();
      image.transpose(1, 2);
    }
  }
}

int main()
{
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    device_type = torch::kCUDA;
  }
  else {
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Net model;
  model.to(device);

  auto train_dataset =
    torch::data::datasets::MNIST(MNIST_PATH, torch::data::datasets::MNIST::Mode::kTrain)
    .map(torch::data::transforms::Stack<>());

  auto train_loader = torch::data::make_data_loader(std::move(train_dataset), BATCH_SIZE);

  auto test_dataset =
    torch::data::datasets::MNIST(MNIST_PATH, torch::data::datasets::MNIST::Mode::kTest)
    .map(torch::data::transforms::Stack<>());

  auto test_loader = torch::data::make_data_loader(std::move(test_dataset), BATCH_SIZE);

  torch::optim::Adam optimizer(model.parameters(), 0.001);

  for (size_t i = 1; i <= EPOCHS; i++) {
    train(i, device, model, *train_loader, optimizer);
  }

  return 0;
}
