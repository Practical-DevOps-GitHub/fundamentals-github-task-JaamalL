#include <math.h>
#include <iostream>

#include <src/neural-network.h>
#include <src/layer.h>
#include <src/activation/sigmoid.h>
#include <src/math/vector/ivector.h>
#include <src/learning/errors/square-error.h>

IOFile NeuralNetwork::mFileStream;

size_t NeuralNetwork::GetLayersCount() const { return this->mLayers.size(); }
IVector<double_t>& NeuralNetwork::GetInputLayer() const { return this->mLayers.front().GetNeurons(); }
IVector<double_t>& NeuralNetwork::GetOutputLayer() const { return this->mLayers.back().GetNeurons(); }

void NeuralNetwork::SetActivation(const size_t layerIdx, const IActivation::Activations activation)
{
    this->mLayers[layerIdx].SetActivation(activation);
}
void NeuralNetwork::SetActivation(const size_t startLayerIdx, const size_t endLayerIdx, const IActivation::Activations activation)
{
    for (size_t i = startLayerIdx; i < endLayerIdx; ++i)
        this->mLayers[i].SetActivation(activation);
}
void NeuralNetwork::SetActivation(const std::vector<IActivation::Activations>& activationArray, const size_t startLayerIdx)
{
    for (size_t i = startLayerIdx; i < activationArray.size(); ++i)
        this->mLayers[i].SetActivation(activationArray[i]);
}

void NeuralNetwork::SetLayers(const std::vector<size_t> &layers) { InstantiateLayers(layers); }
void NeuralNetwork::SetLearning(ILearning &&learning) 
{ 
    this->mLearningImpl = std::move(learning.MakeLearning(&this->mLayers, this->mVectorImpl.get(), this->mMatrixImpl.get(), this->mErrorImpl.get())); 
}
void NeuralNetwork::SetErrors(IErrors &&errors) { this->mErrorImpl = std::move(errors.MakeErrors(&this->mLayers, this->mVectorImpl.get())); }
void NeuralNetwork::SetMatrix(IMatrix<double_t> &&matrix) { this->mMatrixImpl = std::move(matrix.MakeMatrix()); }
void NeuralNetwork::SetVector(IVector<double_t> &&vector) { this->mVectorImpl = std::move(vector.MakeVector()); }



void NeuralNetwork::InstantiateLayers(const std::vector<size_t>& layers)
{
    this->mLayers.emplace_back(Layer(layers.front(), 0, this->mMatrixImpl.get(), this->mVectorImpl.get()));
    for (size_t i = 1; i < layers.size(); ++i)
    {
        Layer layer(layers[i], layers[i - 1], this->mMatrixImpl.get(), this->mVectorImpl.get());
        layer.GetBias().RandD();
        layer.GetWeights().RandD();
        this->mLayers.emplace_back(std::move(layer));
        this->mLayers[i].SetActivation<Sigmoid>();
    }
}
void NeuralNetwork::CreateLayers(const std::vector<size_t>& layers)
{
    this->mLayers.emplace_back(Layer(layers.front(), 0, this->mMatrixImpl.get(), this->mVectorImpl.get()));
    for (size_t i = 1; i < layers.size(); ++i)
    {
        this->mLayers.emplace_back(Layer(layers[i], layers[i - 1], this->mMatrixImpl.get(), this->mVectorImpl.get()));
        this->mLayers[i].SetActivation<Sigmoid>();
    }
}

void NeuralNetwork::MathHandler()
{
    if (std::thread::hardware_concurrency() > 1)
    {
        this->mMatrixImpl = std::make_unique<Matrix<double_t>>(); // MatrixMultithreaded
        this->mVectorImpl = std::make_unique<Vector<double_t>>(); // VectorMultithreaded
    }
    else
    {
        this->mMatrixImpl = std::make_unique<Matrix<double_t>>();
        this->mVectorImpl = std::make_unique<Vector<double_t>>();
    }
}

IVector<double_t>& NeuralNetwork::FeedForward(std::vector<double_t>& input)
{
    mLayers.front().GetNeurons().UseArr(&input[0], mLayers[0].GetNeuronsCount());

    for (size_t i = 1; i < this->mLayers.size(); ++i)
    {
        this->mLayers[i].GetNeurons().Mul(mLayers[i].GetWeights(), this->mLayers[i - 1].GetNeurons());
        this->mLayers[i].GetNeurons().Add(this->mLayers[i].GetBias());
        this->mLayers[i].Activate();
    }

    return this->mLayers.back().GetNeurons();
}

void NeuralNetwork::Learning(std::vector<double_t> input, std::vector<double_t> target)
{
    FeedForward(input);
    this->mLearningImpl->Learning(target);
}

void NeuralNetwork::Save(const std::string fileName) const
{
    mFileStream.Save(this, fileName);
}
void NeuralNetwork::Load(const std::string fileName)
{
    mFileStream.Load(this, fileName);
}
