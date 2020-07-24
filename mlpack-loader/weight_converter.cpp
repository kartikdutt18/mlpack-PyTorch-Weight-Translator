#include<bits/stdc++.h>
#include "models/models/models.hpp"
#include "models/models/darknet/darknet.hpp"
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#define ll long long
using namespace std;

template<
    typename OutputLayer = mlpack::ann::NegativeLogLikelihood<>,
    typename InitializationRule = mlpack::ann::RandomInitialization>
void LoadWeights(mlpack::ann::FFN<OutputLayer, InitializationRule>& model,
                 std::string modelConfigPath)
{
  size_t currentOffset = 0;
  boost::property_tree::ptree xmlFile;
  boost::property_tree::read_xml(modelConfigPath, xmlFile);
  boost::property_tree::ptree modelConfig = xmlFile.get_child("model");
  BOOST_FOREACH(boost::property_tree::ptree::value_type const& layer, modelConfig)
  {
    std::cout << currentOffset << std::endl;
    // Load Weights.
    if (layer.second.get_child("has_weights").data() != "0")
    {
      arma::mat weights;
      mlpack::data::Load("./../" + layer.second.get_child("weight_csv").data(), weights);
      model.Parameters()(arma::span(currentOffset, currentOffset + weights.n_elem - 1),
          arma::span()) = weights.t();
      currentOffset += weights.n_elem;
    }
    else
    {
      currentOffset += std::stoi(layer.second.get_child("weight_offset").data());
    }

    // Load Biases.
    if (layer.second.get_child("has_bias").data() != "0")
    {
      arma::mat bias;
      mlpack::data::Load("./../" + layer.second.get_child("bias_csv").data(), bias);
      model.Parameters()(arma::span(currentOffset, currentOffset + bias.n_elem - 1),
                         arma::span()) = bias.t();
      currentOffset += bias.n_elem;
    }
    else
    {
      currentOffset += std::stoi(layer.second.get_child("bias_offset").data());
    }
  }
  std::cout << currentOffset << std::endl;
}

template<typename LayerType = mlpack::ann::FFN<>>
LayerType LoadRunningMeanAndVariance(LayerType&& baseLayer, size_t i = 0)
{
  while (i < baseLayer.Model().size() && !batchNormRunningMean.empty())
  {
    if (baseLayer.Model()[i].type() == typeid(new mlpack::ann::Sequential<>()))
    {
      std::cout << "Sequential Layer. " << i << std::endl;
      LoadRunningMeanAndVariance<mlpack::ann::Sequential<>>(std::move(baseLayer.Model()[i]));
    }

    if (!batchNormRunningMean.empty() &&
        baseLayer.Model()[i].type() == typeid(new mlpack::ann::BatchNorm<>()))
    {
      std::cout << "BATCHNORM Layer " << i << std::endl;
      arma::mat runningMean;
      mlpack::data::Load(batchNormRunningMean.front(), runningMean);
      batchNormRunningMean.pop();
     // baseLayer.Model()[i].TrainingMean() = runningMean;
     // baseLayer.Model()[i].TrainingMean().print();
    }

    i++;
  }
  return baseLayer;
}

int main(int argc, char **argv)
{
  if (argc < 1)
  {
    std::cout << "The methods requires atleast one model name!" << std::endl;
    return 1;
  }

  for (int i = 0; i < argc; ++i)
  {
    if (strncmp(argv[i], "darknet19", 9))
    {
      mlpack::ann::FFN<> model2;
      model2.ResetParameters();
      model2.Parameters() = arma::mat(20849576, 1);
      LoadWeights<>(model2, "./../cfg/darknet19.xml");
    }
  }

  return 0;
}