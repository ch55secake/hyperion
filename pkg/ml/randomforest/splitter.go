package randomforest

import (
	types "github.com/ch55secake/hyperion/pkg/data"
	"math/rand/v2"
	"sort"
)

func findBestSplit(data []types.Features, maxFeatures int) (int, float64, float64) {
	bestGini := 1.0
	bestFeature := 0
	bestThreshold := 0.0

	// Randomly select features
	numFeatures := 18
	features := rand.Perm(numFeatures)[:maxFeatures]

	for _, featureIdx := range features {
		values := make([]float64, len(data))
		for i, f := range data {
			values[i] = types.GetFeatureValue(f, featureIdx)
		}

		sort.Float64s(values)

		// Try more split points
		for q := 1; q <= 9; q++ {
			idx := len(values) * q / 10
			if idx >= len(values) || idx == 0 {
				continue
			}
			threshold := values[idx]

			left, right := splitData(data, featureIdx, threshold)
			if len(left) < 5 || len(right) < 5 {
				continue
			}

			gini := calculateGini(left, right)
			if gini < bestGini {
				bestGini = gini
				bestFeature = featureIdx
				bestThreshold = threshold
			}
		}
	}

	return bestFeature, bestThreshold, bestGini
}

func splitData(data []types.Features, featureIdx int, threshold float64) ([]types.Features, []types.Features) {
	var left, right []types.Features

	for _, f := range data {
		if types.GetFeatureValue(f, featureIdx) <= threshold {
			left = append(left, f)
		} else {
			right = append(right, f)
		}
	}

	return left, right
}

func calculateGini(left, right []types.Features) float64 {
	totalSize := float64(len(left) + len(right))
	giniLeft := giniImpurity(left)
	giniRight := giniImpurity(right)

	return (float64(len(left))/totalSize)*giniLeft + (float64(len(right))/totalSize)*giniRight
}

func giniImpurity(data []types.Features) float64 {
	if len(data) == 0 {
		return 0
	}

	ones := 0
	for _, f := range data {
		if f.Label == 1 {
			ones++
		}
	}

	p1 := float64(ones) / float64(len(data))
	p0 := 1.0 - p1

	return 1.0 - (p1*p1 + p0*p0)
}
