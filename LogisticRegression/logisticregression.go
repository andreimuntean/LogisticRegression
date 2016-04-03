package main

import (
    "fmt"
    "math"
    "math/rand"
)

func getSigmoid(value float64) float64 {
    return 1 / (1 + math.Exp(-value))
}

func getActivation(weights []float64, x []float64) float64 {
    var activation = 0.0

    for i, _ := range x {
        activation += weights[i] * x[i]
    }

    return getSigmoid(activation)
}

func getLoss(weights []float64, x []float64, y int) (float64, []float64) {
    var loss = 0.0
    var gradients = make([]float64, len(weights))

    // Computes the activation.
    var activation = getActivation(weights, x)
    
    // Computes the loss using maximum log-likelihood.
    if y == 0 {
        loss -= math.Log(1 - activation)
    } else {
        loss -= math.Log(activation)
    }

    // Computes the gradients.
    for i, _ := range gradients {
        gradients[i] += x[i] * (activation - float64(y))
    }

    return loss, gradients
}

func getBatchLoss(weights []float64, x [][]float64, y []int) (float64, []float64) {
    var batchLoss = 0.0
    var batchGradients = make([]float64, len(weights))

    for i, _ := range x {
        // Computes the loss and gradients for this example.
        var loss, gradients = getLoss(weights, x[i], y[i])

        // Accumulates and averages the values.
        batchLoss += loss / float64(len(x))

        for j, _ := range gradients {
            batchGradients[j] += gradients[j] / float64(len(x))
        }
    }

    return batchLoss, batchGradients
}

func main() {
    var x = [][]float64 {
        { 1.0, 0.6 },
        { 1.0, 0.2 },
        { 1.0, 1.0 },
        { 1.0, 0.0 },
        { 1.0, 0.8 },
        { 1.0, 0.49 },
        { 1.0, 0.51 }}
    var y = []int { 1, 0, 1, 0, 1, 0, 1 }

    // Configures the hyperparameters.
    const learningRate = 1.0
    const maxIterations = 1000000
    const epsilon = 1e-3

    // Generates random weights.
    var weights = []float64 { rand.Float64(), rand.Float64() }

    // Trains the classifier.
    for epoch := 0; epoch < maxIterations; epoch++ {
        var loss, gradients = getBatchLoss(weights, x, y)

        if loss <= epsilon {
            fmt.Printf("Converged after %d iterations.\n", epoch)
            break
        }

        for i, _ := range weights {
            weights[i] -= learningRate * gradients[i]
        }
    }

    var loss, _ = getBatchLoss(weights, x, y)

    fmt.Printf("Loss: %f.\n", loss)
}