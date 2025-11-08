package randomforest

import (
	"fmt"
	"math/rand"
	"time"

	util "github.com/ch55secake/hyperion/hyperion-go/pkg"
	types "github.com/ch55secake/hyperion/hyperion-go/pkg/data"
)

// Train with improved Random Forest
func Train(ts *types.TradingStrategy, numTrees, maxDepth, minSample int) {
	ts.TrainSize = int(float64(len(ts.Features)) * 0.7)
	trainData := ts.Features[:ts.TrainSize]

	// Check label distribution
	ones := 0
	for _, f := range trainData {
		if f.Label == 1 {
			ones++
		}
	}
	fmt.Printf("Training data - Up: %d (%.1f%%), Down: %d (%.1f%%)\n",
		ones, float64(ones)/float64(len(trainData))*100,
		len(trainData)-ones, float64(len(trainData)-ones)/float64(len(trainData))*100)

	ts.Model = &types.RandomForest{
		NumTrees:    numTrees,
		MaxDepth:    maxDepth,
		MinSample:   minSample,
		MaxFeatures: 8, // sqrt(18) ≈ 4, but use more for better splits
		Trees:       make([]*types.DecisionTree, numTrees),
	}

	fmt.Printf("Training Random Forest with %d trees...\n", numTrees)

	rand.Seed(time.Now().UnixNano())

	for i := 0; i < numTrees; i++ {
		sample := bootstrapSample(trainData)
		ts.Model.Trees[i] = buildTree(sample, 0, maxDepth, minSample, ts.Model.MaxFeatures)
		if (i+1)%5 == 0 {
			util.PrintSimpleProgress(fmt.Sprintf("Training %d trees", numTrees), i+1, numTrees)
		}
	}

	fmt.Println("\nTraining complete!")
}

func bootstrapSample(data []types.Features) []types.Features {
	sample := make([]data.Features, len(data))
	for i := 0; i < len(data); i++ {
		idx := rand.Intn(len(data))
		sample[i] = data[idx]
	}
	return sample
}
