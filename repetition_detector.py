#!/usr/bin/env python3
"""	Script to detect repetitive patterns in LLM benchmark results,
	identifying cases where models get stuck in loops.
"""

import json
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Set


def find_repeating_patterns(text: str, min_length: int = 10, 
                          min_occurrences: int = 3) -> List[Tuple[str, int, List[int]]]:
	"""	Find the longest repeating patterns in text by expanding from 
		shorter patterns.
		
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
		including overlapping occurrences.
	"""
	positions = []
	start = 0
	while True:
		pos = text.find(pattern, start)
		if pos == -1:
			break
		positions.append(pos)
		start = pos + 1  # Allow overlapping matches
	return positions


def analyze_json_file(filepath: str, min_length: int = 10, 
                     min_occurrences: int = 3) -> Dict:
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
			# Add pattern information
			most_frequent_pattern = patterns[0]
			entry_data['most_frequent_pattern'] = {
				'text': most_frequent_pattern[0],
				'length': len(most_frequent_pattern[0]),
				'occurrences': most_frequent_pattern[1],
				'positions': most_frequent_pattern[2]
			}
			entry_data['all_patterns'] = [
				{
					'text': p[0],
					'length': len(p[0]),
					'occurrences': p[1],
					'positions': p[2]
				}
				for p in patterns
			]
		
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
				preview = most_frequent['text'][:150]
				print(f"  Preview: '{preview}{'...' if len(most_frequent['text']) > 150 else ''}'")
		
		if len(entries_with_patterns) > 3:
			print(f"\n... and {len(entries_with_patterns) - 3} more entries with repetition")


def main():
	parser = argparse.ArgumentParser(
		description="Detect repetitive patterns in LLM benchmark results"
	)
	parser.add_argument('json_file', help='Path to JSON file containing results')
	parser.add_argument('--min-length', type=int, default=50,
						help='Minimum pattern length to search for (default: 50)')
	parser.add_argument('--min-occurrences', type=int, default=5,
						help='Minimum number of repetitions required (default: 5)')
	parser.add_argument('--output', '-o', help='Output detailed results to JSON file')
	
	args = parser.parse_args()
	
	try:
		results = analyze_json_file(args.json_file, args.min_length, args.min_occurrences)
		print_analysis_summary(results)
		
		if args.output:
			with open(args.output, 'w', encoding='utf-8') as f:
				json.dump(results, f, indent=2, ensure_ascii=False)
			print(f"\nDetailed results saved to: {args.output}")
			
	except Exception as e:
		print(f"Error: {e}")
		return 1
	
	return 0


if __name__ == '__main__':
	exit(main())