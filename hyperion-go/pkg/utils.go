package pkg

import "fmt"

// PrintSimpleProgress prints a simple progress percentage rather than creating new lines
func PrintSimpleProgress(msg string, current, total int) {
	fmt.Printf("\r%s: %d/%d [%.1f%%]",
		msg,
		current,
		total,
		float64(current)*100/float64(total),
	)
}
