package randomforest

import (
	types "github.com/ch55secake/hyperion/hyperion-go/pkg/data"
)

func buildTree(data []types.Features, depth, maxDepth, minSample, maxFeatures int) *types.DecisionTree {
	if depth >= maxDepth || len(data) < minSample {
		return &data.DecisionTree{
			IsLeaf:     true,
			Prediction: calculateMajorityClass(data),
		}
	}

	// Check if all same label
	allSame := true
	firstLabel := data[0].Label
	for _, f := range data {
		if f.Label != firstLabel {
			allSame = false
			break
		}
	}

	if allSame {
		return &data.DecisionTree{
			IsLeaf:     true,
			Prediction: float64(firstLabel),
		}
	}

	bestFeature, bestThreshold, bestGini := findBestSplit(data, maxFeatures)

	if bestGini >= 0.49 { // No significant improvement
		return &data.DecisionTree{
			IsLeaf:     true,
			Prediction: calculateMajorityClass(data),
		}
	}

	left, right := splitData(data, bestFeature, bestThreshold)

	if len(left) == 0 || len(right) == 0 {
		return &data.DecisionTree{
			IsLeaf:     true,
			Prediction: calculateMajorityClass(data),
		}
	}

	return &data.DecisionTree{
		FeatureIdx: bestFeature,
		Threshold:  bestThreshold,
		Left:       buildTree(left, depth+1, maxDepth, minSample, maxFeatures),
		Right:      buildTree(right, depth+1, maxDepth, minSample, maxFeatures),
		IsLeaf:     false,
	}
}

func calculateMajorityClass(data []types.Features) float64 {
	ones := 0
	for _, f := range data {
		if f.Label == 1 {
			ones++
		}
	}

	if ones > len(data)/2 {
		return 1.0
	}
	return 0.0
}
