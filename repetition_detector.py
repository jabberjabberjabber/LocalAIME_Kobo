#!/usr/bin/env python3
"""	Script to detect repetitive patterns in LLM benchmark results,
	identifying cases where models get stuck in loops.
	
	Key improvements:
	- Uses non-overlapping search to avoid inflated occurrence counts
	- Prioritizes frequency over pattern length
	- Detects internal repetition to find atomic repetitive units
	- Optional whitespace normalization for semantic pattern analysis
	- Always outputs detailed JSON results with smart default naming
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Set


def find_repeating_patterns(text: str, min_length: int = 10, 
                          min_occurrences: int = 3) -> List[Tuple[str, int, List[int]]]:
	"""	Find repeating patterns in text, prioritizing frequency over length.
		Uses non-overlapping search to avoid artificial count inflation.
		
		Args:
			text: Input text to analyze
			min_length: Minimum substring length to start searching
			min_occurrences: Minimum number of times pattern must repeat
			
		Returns:
			List of (pattern, count, positions) tuples, sorted by occurrence count desc
	"""
	if len(text) < min_length * min_occurrences:
		return []
	
	# Track patterns we've already found to avoid redundant work
	found_patterns = set()
	results = []
	
	# Start with minimum length and work our way through the text
	for start_pos in range(len(text) - min_length + 1):
		# Skip if we've already found a pattern starting here
		if start_pos in found_patterns:
			continue
			
		current_length = min_length
		best_pattern = None
		best_count = 0
		best_positions = []
		
		# Expand pattern length while it still repeats enough times
		while start_pos + current_length <= len(text):
			pattern = text[start_pos:start_pos + current_length]
			positions = find_all_occurrences(text, pattern)
			
			if len(positions) >= min_occurrences:
				# This length still repeats enough - save it and try longer
				best_pattern = pattern
				best_count = len(positions)
				best_positions = positions.copy()
				current_length += 1
			else:
				# Can't expand further while maintaining repetition count
				break
		
		# If we found a valid pattern, record it
		if best_pattern:
			results.append((best_pattern, best_count, best_positions))
			# Mark all positions where this pattern occurs
			for pos in best_positions:
				for i in range(pos, pos + len(best_pattern)):
					found_patterns.add(i)
	
	# Sort by occurrence count (most frequent first), then by pattern length
	results.sort(key=lambda x: (-x[1], -len(x[0])))
	return results


def find_all_occurrences(text: str, pattern: str) -> List[int]:
	"""	Find all starting positions where pattern occurs in text,
		using non-overlapping occurrences only.
	"""
	positions = []
	start = 0
	while True:
		pos = text.find(pattern, start)
		if pos == -1:
			break
		positions.append(pos)
		start = pos + len(pattern)  # Non-overlapping matches only
	return positions


def find_internal_repetition(pattern: str) -> Dict:
	"""	Check if a pattern is composed of a smaller repeating unit.
		
		Args:
			pattern: The pattern to analyze
			
		Returns:
			Dict with 'unit', 'repetitions' if internal repetition found, None otherwise
	"""
	for unit_length in range(1, len(pattern) // 2 + 1):
		unit = pattern[:unit_length]
		repetitions = len(pattern) / unit_length
		
		# Check if repeating this unit recreates the pattern
		if unit * int(repetitions) == pattern[:int(repetitions) * unit_length]:
			# Check if remaining chars match start of unit (for partial repetitions)
			remainder = len(pattern) % unit_length
			if remainder == 0 or pattern[-remainder:] == unit[:remainder]:
				return {
					'unit': unit,
					'repetitions': repetitions,
					'is_exact': remainder == 0
				}
	return None


def analyze_json_file(filepath: str, min_length: int = 10, 
                     min_occurrences: int = 3, decompose: bool = False, 
                     normalize: bool = False) -> Dict:
	"""	Load JSON file and analyze each result's response_text for 
		repetitive patterns.
	"""
	try:
		with open(filepath, 'r', encoding='utf-8') as f:
			data = json.load(f)
	except FileNotFoundError:
		raise FileNotFoundError(f"Could not find file: {filepath}")
	except json.JSONDecodeError as e:
		raise ValueError(f"Invalid JSON in file {filepath}: {e}")
	
	if 'results' not in data:
		raise ValueError("JSON file must contain a 'results' key")
	
	analysis_results = {
		'total_entries': len(data['results']),
		'entries_with_repetition': 0,
		'correct_answers': 0,
		'incorrect_answers': 0,
		'repetition_by_correctness': {
			'correct_with_repetition': 0,
			'incorrect_with_repetition': 0
		},
		'analysis_settings': {
			'min_length': min_length,
			'min_occurrences': min_occurrences,
			'decompose': decompose,
			'normalize': normalize
		},
		'entries': []
	}
	
	for idx, result in enumerate(data['results']):
		if 'response_text' not in result:
			print(f"Warning: Entry {idx} missing 'response_text' key")
			continue
		
		# Check correctness
		is_correct = None
		if 'expected_int' in result and 'response_int' in result:
			is_correct = result['expected_int'] == result['response_int']
			if is_correct:
				analysis_results['correct_answers'] += 1
			else:
				analysis_results['incorrect_answers'] += 1
		
		response_text = result['response_text']
		
		# Normalize
		if normalize:
			response_text = re.sub(r'\s+', '', response_text).lower()
		
		patterns = find_repeating_patterns(response_text, min_length, min_occurrences)
		
		has_repetition = len(patterns) > 0
		if has_repetition:
			analysis_results['entries_with_repetition'] += 1
			if is_correct is not None:
				if is_correct:
					analysis_results['repetition_by_correctness']['correct_with_repetition'] += 1
				else:
					analysis_results['repetition_by_correctness']['incorrect_with_repetition'] += 1
		
		# Create entry for this result
		entry_data = {
			'entry_index': idx,
			'expected_int': result.get('expected_int'),
			'response_int': result.get('response_int'),
			'is_correct': is_correct,
			'text_length': len(response_text),
			'has_repetition': has_repetition,
			'patterns_found': len(patterns)
		}
		
		if patterns:
			# Add pattern information, including decomposition analysis if requested
			most_frequent_pattern = patterns[0]
			
			entry_data['most_frequent_pattern'] = {
				'text': most_frequent_pattern[0],
				'length': len(most_frequent_pattern[0]),
				'occurrences': most_frequent_pattern[1],
				'positions': most_frequent_pattern[2]
			}
			
			# Add decomposition info if requested and found
			if decompose:
				internal_rep = find_internal_repetition(most_frequent_pattern[0])
				if internal_rep:
					entry_data['most_frequent_pattern']['internal_repetition'] = {
						'atomic_unit': internal_rep['unit'],
						'unit_repetitions': internal_rep['repetitions'],
						'is_exact_repetition': internal_rep['is_exact'],
						'total_atomic_occurrences': most_frequent_pattern[1] * internal_rep['repetitions']
					}
			
			entry_data['all_patterns'] = []
			for p in patterns:
				pattern_info = {
					'text': p[0],
					'length': len(p[0]),
					'occurrences': p[1],
					'positions': p[2]
				}
				
				# Check each pattern for internal repetition if requested
				if decompose:
					internal = find_internal_repetition(p[0])
					if internal:
						pattern_info['internal_repetition'] = {
							'atomic_unit': internal['unit'],
							'unit_repetitions': internal['repetitions'],
							'is_exact_repetition': internal['is_exact'],
							'total_atomic_occurrences': p[1] * internal['repetitions']
						}
				
				entry_data['all_patterns'].append(pattern_info)
		
		analysis_results['entries'].append(entry_data)
	
	return analysis_results


def print_analysis_summary(results: Dict):
	"""	Print a human-readable summary of the analysis results.
	"""
	print(f"\n=== LLM Repetition Analysis Summary ===")
	print(f"Total entries analyzed: {results['total_entries']}")
	print(f"Entries with repetitive patterns: {results['entries_with_repetition']}")
	
	# Correctness statistics
	total_with_answers = results['correct_answers'] + results['incorrect_answers']
	if total_with_answers > 0:
		print(f"\n=== Correctness Analysis ===")
		print(f"Entries with answer comparison: {total_with_answers}")
		print(f"Correct answers: {results['correct_answers']} ({results['correct_answers']/total_with_answers*100:.1f}%)")
		print(f"Incorrect answers: {results['incorrect_answers']} ({results['incorrect_answers']/total_with_answers*100:.1f}%)")
		
		print(f"\n=== Repetition vs Correctness ===")
		correct_rep = results['repetition_by_correctness']['correct_with_repetition']
		incorrect_rep = results['repetition_by_correctness']['incorrect_with_repetition']
		
		if results['correct_answers'] > 0:
			print(f"Correct answers with repetition: {correct_rep}/{results['correct_answers']} ({correct_rep/results['correct_answers']*100:.1f}%)")
		if results['incorrect_answers'] > 0:
			print(f"Incorrect answers with repetition: {incorrect_rep}/{results['incorrect_answers']} ({incorrect_rep/results['incorrect_answers']*100:.1f}%)")
	
	if results['entries_with_repetition'] == 0:
		print("No repetitive patterns found!")
		return
	
	print(f"\nOverall repetition rate: {results['entries_with_repetition']/results['total_entries']*100:.1f}%")
	
	# Show examples of entries with repetition
	entries_with_patterns = [e for e in results['entries'] if e['has_repetition']]
	if entries_with_patterns:
		print(f"\n=== Examples of Repetitive Entries ===")
		for i, entry in enumerate(entries_with_patterns[:3]):  # Show first 3 examples
			print(f"\nEntry {entry['entry_index']}:")
			if entry['is_correct'] is not None:
				correctness = "✓ Correct" if entry['is_correct'] else "✗ Incorrect"
				print(f"  Answer: {correctness} (expected: {entry['expected_int']}, got: {entry['response_int']})")
			print(f"  Text length: {entry['text_length']} characters")
			print(f"  Patterns found: {entry['patterns_found']}")
			
			if 'most_frequent_pattern' in entry:
				most_frequent = entry['most_frequent_pattern']
				print(f"  Most frequent pattern: {most_frequent['length']} chars, {most_frequent['occurrences']} times")
				
				# Show decomposition info if available
				if 'internal_repetition' in most_frequent:
					internal = most_frequent['internal_repetition']
					exactness = "exact" if internal['is_exact_repetition'] else "partial"
					print(f"  → Atomic unit: '{internal['atomic_unit']}' × {internal['unit_repetitions']:.1f} ({exactness})")
					print(f"  → Total atomic occurrences: {internal['total_atomic_occurrences']:.1f}")
				
				preview = most_frequent['text'][:150]
				print(f"  Preview: '{preview}{'...' if len(most_frequent['text']) > 150 else ''}'")
		
		if len(entries_with_patterns) > 3:
			print(f"\n... and {len(entries_with_patterns) - 3} more entries with repetition")


def main():
	parser = argparse.ArgumentParser(
		description="Detect repetitive patterns in LLM benchmark results"
	)
	parser.add_argument('json_file', help='Path to JSON file containing results')
	parser.add_argument('--min-length', type=int, default=25,
						help='Minimum pattern length to search for (default: 25)')
	parser.add_argument('--min-occurrences', type=int, default=5,
						help='Minimum number of repetitions required (default: 5)')
	parser.add_argument('--decompose', action='store_true',
						help='Analyze patterns for internal repetition (finds atomic units)')
	parser.add_argument('--normalize', action='store_true',
						help='Strip whitespace and convert to lowercase to focus on semantic patterns')
	parser.add_argument('--output', '-o', help='Output JSON file (default: input filename with -repeats suffix)')
	
	args = parser.parse_args()
	
	# Generate default output filename if not provided
	if not args.output:
		input_path = Path(args.json_file)
		output_filename = f"{input_path.stem}-repeats{input_path.suffix}"
		args.output = input_path.parent / output_filename
	
	try:
		results = analyze_json_file(args.json_file, args.min_length, args.min_occurrences, 
								   args.decompose, args.normalize)
		print_analysis_summary(results)
		
		# Always output JSON results
		with open(args.output, 'w', encoding='utf-8') as f:
			json.dump(results, f, indent=2, ensure_ascii=False)
		print(f"\nDetailed results saved to: {args.output}")
			
	except Exception as e:
		print(f"Error: {e}")
		return 1
	
	return 0


if __name__ == '__main__':
	exit(main())