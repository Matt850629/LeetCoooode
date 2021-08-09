package test_project;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Stack;

public class Leetcode {

	public Leetcode() {
		super();
	}

//	public int lengthOfLongestSubstring(String s) {
//
//		int[] last = new int[128];
//		Arrays.fill(last, -1);
//		int start = 0;
//		int ans = 0;
//		for (int i = 0; i < s.length(); ++i) {
//			if (last[s.charAt(i)] != -1)
//				start = Math.max(start, last[s.charAt(i)] + 1);
//			last[s.charAt(i)] = i;
//			ans = Math.max(ans, i - start + 1);
//		}
//		return ans;
//	}
	// 3.Longest Substring Without Repeating
	public static int lengthOfLongestSubstring(String s) {
		HashMap<Character, Integer> seen = new HashMap<>();
		int maximum_length = 0;
		// starting the inital point of window to index 0
		int start = 0;
		for (int end = 0; end < s.length(); end++) {
			// Checking if we have already seen the element or not
			if (seen.containsKey(s.charAt(end))) {
				// If we have seen the number, move the start pointer
				// to position after the last occurrence
				start = Math.max(start, seen.get(s.charAt(end)) + 1);

			}

			// Updating the last seen value of the character
			seen.put(s.charAt(end), end);
			maximum_length = Math.max(maximum_length, end - start + 1);
		}
		return maximum_length;
	}

	// 5.Longest Palindromic Substring
	public String longestPalindrome(String s) {
		int len = 0;
		int len2 = 0;
		String palindrome = "";
		for (int i = 0; i < s.length(); i++) {
			len = isPalidromic(s, i, i).length();
			if (len > palindrome.length())
				palindrome = isPalidromic(s, i, i);

			len2 = isPalidromic(s, i, i + 1).length();
			if (len2 > palindrome.length())
				palindrome = isPalidromic(s, i, i + 1);

		}
		return palindrome;
	}

	private String isPalidromic(String s, int l, int r) {
		while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
			l--;
			r++;
		}
		return s.substring(l + 1, r);
	}

	// 6.Zigzag Conversion
	public static String ZigzagConversion(String s, int numRows) {
		char[] str = s.toCharArray();
		String rowStr[] = new String[numRows];
		Arrays.fill(rowStr, "");
		int rows = 0;
		boolean down = true;
		String result = "";
		if (numRows == 1) {
			return s;
		}

		for (int i = 0; i < s.length(); i++) {
			rowStr[rows] += str[i];
			if (rows == numRows - 1)
				down = false;
			else if (rows == 0)
				down = true;

			if (down)
				rows++;
			else
				rows--;

		}

		for (int i = 0; i < rowStr.length; i++) {
			result += rowStr[i];
		}
		return result;
	}

	// 7.Reverse Integer
	public static int reverse(int x) {
		long res = 0;

		while (x != 0) {

			res = res * 10 + x % 10;
			x /= 10;
			if (res > Integer.MAX_VALUE)
				return 0;
			if (res < Integer.MIN_VALUE)
				return 0;
		}
		return (int) res;
	}

	// 12.Integer to Roman
	public static String intToRoman(int num) {
		StringBuilder sb = new StringBuilder();
		String[] str = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };
		int[] numeric = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
		for (int i = 0; i < numeric.length; i++) {
			while (num >= numeric[i]) {
				num -= numeric[i];
				sb.append(str[i]);
			}
		}
		return sb.toString();
	}

	// 13.Roman to Integer
	public static int romanToInt(String s) {
		Map<Character, Integer> romanMap = new HashMap<>();
		// Fill the map
		romanMap.put('I', 1);
		romanMap.put('V', 5);
		romanMap.put('X', 10);
		romanMap.put('L', 50);
		romanMap.put('C', 100);
		romanMap.put('D', 500);
		romanMap.put('M', 1000);
		var n = s.length();
		// start from back number
		var num = romanMap.get(s.charAt(n - 1));
		for (int i = n - 2; i >= 0; i--) {
			if (romanMap.get(s.charAt(i)) >= romanMap.get(s.charAt(i + 1)))
				num += romanMap.get(s.charAt(i));
			else
				num -= romanMap.get(s.charAt(i));
		}
		return num;
	}

	// 14.Longest Common Prefix
	public static String longestCommonPrefix(String[] args) {
		if (args == null || args.length == 0)
			return "";

		String st = new String();
		for (int i = 0; i < args[0].length(); i++) {
			var ch = args[0].charAt(i);
			for (int j = 1; j < args.length; j++) {
				if (i >= args[j].length() || ch != args[j].charAt(i))
					return st;
			}
			st += ch;
		}
		return st;
	}

	// 15.3Sum
	public static List<List<Integer>> threeSum(int[] nums) {
		Arrays.sort(nums);
		List<List<Integer>> ans = new LinkedList<>();

		var n = nums.length;
		for (int i = 0; i < n; i++) {
			if (i > 0 && nums[i] == nums[i - 1])
				continue;
			var j = i + 1;
			var k = n - 1;
			while (j < k) {
				if (nums[k] + nums[j] < -nums[i])
					j++;
				else if (nums[k] + nums[j] > -nums[i])
					k--;
				else {
					ans.add(Arrays.asList(nums[k], nums[i], nums[j]));
					while (j < k && nums[j] == nums[j + 1])
						j++;
					while (j < k && nums[k] == nums[k - 1])
						k--;
					j++;
					k--;
				}
			}
		}
		return ans;
	}

	// 16.3Sum closest
	public static int threeSumClosest(int[] nums, int target) {
		int closet = nums[0] + nums[1] + nums[2];
		int diff = Math.abs(target - closet);
		Arrays.sort(nums);
		for (int i = 0; i < nums.length; i++) {
			int l = i + 1;
			int r = nums.length - 1;
			while (r > l) {
				int newDiff = Math.abs(target - (nums[l] + nums[r] + nums[i]));
				int sum = nums[l] + nums[i] + nums[r];
				if (diff > newDiff) {
					diff = newDiff;
					closet = sum;
				}
				if (target > sum)
					l++;
				else
					r--;
			}
		}
		return closet;
	}

	static String[] board = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tvu", "wxyz" };

	// 17.Letter Combinations of a Phone Number
	// Use Recursion dfs(digits, digits index , temp space , result)
	public static List<String> letterCombinations(String digits) {
		List<String> result = new ArrayList<>();
		StringBuilder sb = new StringBuilder();
		int index = 0;
		if (digits.length() == 0)
			return result;
		dfs(digits, index, sb, result);
		return result;
	}

	private static void dfs(String digits, int index, StringBuilder sb, List<String> result) {
		if (digits.length() == index) {
			result.add(sb.toString());
			return;
		}
		String letter = board[digits.charAt(index) - '0'];
		for (int i = 0; i < letter.length(); i++) {
			sb.append(letter.charAt(i));
			dfs(digits, index + 1, sb, result);
			sb.deleteCharAt(sb.length() - 1);
		}

	}

	// 18.4Sum
	public static List<List<Integer>> fourSum(int[] nums, int target) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		Arrays.sort(nums);
		int n = nums.length;

		for (int i = 0; i < n - 3; i++) {

			if (i > 0 && nums[i] == nums[i - 1])
				continue;

			for (int j = i + 1; j < n - 2; j++) {
				if (j > i + 1 && nums[j] == nums[j - 1])
					continue;

				int l = j + 1;
				int r = n - 1;
				while (l < r) {
					var sum = nums[i] + nums[j] + nums[l] + nums[r];
					if (sum == target) {
						res.add(Arrays.asList(nums[i], nums[j], nums[l], nums[r]));
						while (l < r && nums[l] == nums[l + 1])
							l++;
						while (l < r && nums[r] == nums[r - 1])
							r--;
						l++;
						r--;
					} else if (sum < target)
						l++;
					else
						r--;

				}
			}

		}
		return res;

	}

	// 19.Remove Nth Node From End of List
	// use two pointer two count distance, when first point is null
	// remove the target node
	public static ListNode removeNthFromEnd(ListNode head, int n) {
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode first = dummy;
		ListNode second = dummy;
		for (int i = 0; i <= n; i++)
			first = first.next;

		while (first != null) {
			first = first.next;
			second = second.next;
		}
		second.next = second.next.next;

		return dummy.next;

	}

	// 20.Valid Parentheses
	// use stack to store character , if this character pairs the last character in
	// stack, if true we remove and then if stack is empty means target is valid
	// Parentheses
	public static boolean isValid(String s) {
		Stack<Character> stack = new Stack<>();
		for (Character ch : s.toCharArray()) {
			if (ch == '{' || ch == '[' || ch == '(')
				stack.push(ch);
			else if (ch == ')' && !stack.isEmpty() && '(' == stack.peek())
				stack.pop();
			else if (ch == '}' && !stack.isEmpty() && '{' == stack.peek())
				stack.pop();
			else if (ch == ']' && !stack.isEmpty() && '[' == stack.peek())
				stack.pop();
			else
				return false;

		}
		return stack.isEmpty();
	}

	// 21.Merge Two Sorted Lists
	// if one list is null return another, compare the two nodes value ,use
	// recursion to lookup all values
	public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		if (l1 == null)
			return l2;
		if (l2 == null)
			return l1;

		if (l1.value < l2.value) {
			l1.next = mergeTwoLists(l1.next, l2);
			return l1;
		} else {
			l2.next = mergeTwoLists(l1, l2.next);
			return l2;
		}

	}

	public static ListNode mergeTwoLists2(ListNode l1, ListNode l2) {
		ListNode dummy = new ListNode(0);
		var cur = dummy;
		while (l1 != null && l2 != null) {
			if (l1.value < l2.value) {
				cur.next = l1;
				l1 = l1.next;
			} else {
				cur.next = l2;
				l2 = l2.next;
			}
			cur = cur.next;
		}
		cur.next = (l1 != null) ? l1 : l2;
		return dummy.next;
	}

	// 1721.Swapping Nodes in a Linked List
	public static ListNode swapNodes(ListNode head, int k) {
		// we notice two nodes distance to sides would be equal
		// we start from head first
		int steps = k - 1;
		ListNode fNode = head;
		while (steps > 0) {
			fNode = fNode.next;
			steps--;
		}

		ListNode f = fNode;
		ListNode sNode = head;
		while (f.next != null) {
			sNode = sNode.next;
			f = f.next;
		}

		int temp = fNode.value;
		fNode.value = sNode.value;
		sNode.value = temp;

		return head;
	}

	// 24.Swap Nodes in Pairs
	// we only change the pointers for each nodes, rather than swap them.
	public static ListNode swapPairs(ListNode head) {
		ListNode dummy = new ListNode(0);
		ListNode cur = dummy;
		dummy.next = head;

		while (cur.next != null && cur.next.next != null) {
			var first = cur.next;
			var second = cur.next.next;
			first.next = second.next;
			second.next = first;
			cur.next = second;
			cur = cur.next.next;
		}
		return dummy.next;

	}

	// 27.Remove Element ,use two pointers
	public static int moveElement(int[] nums, int val) {
		int i = 0;
		for (int j = 0; j < nums.length; j++) {
			if (nums[j] == val)
				continue;
			nums[i] = nums[j];
			i++;
		}
		return i;
	}

	// 28.Implement strStr()
	// compare two string with two loop, increment the point in target
	public static int strStr(String haystack, String needle) {
		int n = needle.length();
		int m = haystack.length();
		for (int i = 0; i + n <= m; i++) {
			boolean flag = true;
			for (int j = 0; j < n; j++) {
				if (haystack.charAt(i + j) != needle.charAt(j)) {
					flag = false;
					break;
				}
			}
			if (flag)
				return i;
		}
		return -1;
	}

	// 29.Divide Two Integers
	// Linear search, confirm the result will be + or -;
	public static int divide(int dividend, int divisor) {

		if (dividend == Integer.MIN_VALUE && divisor == -1)
			return Integer.MAX_VALUE;
		if (dividend == Integer.MIN_VALUE && divisor == 1)
			return Integer.MIN_VALUE;
		int result = 0;
		var positive = (dividend < 0) == (divisor < 0);
		long d = dividend;
		long s = divisor;
		d = Math.abs(d);
		s = Math.abs(s);
		while (d >= s) {
			result += 1;
			d -= s;
		}
		if (positive)
			return (int) Math.min(result, Integer.MAX_VALUE);
		else
			return (int) Math.max(-result, Integer.MIN_VALUE);

	}

	// 31.Next Permutation
	public static void nextPremutation(int nums[]) {
		// corner case
		// {1,2,4,6,5,3}
		// i j
		if (nums == null || nums.length == 0)
			return;
		int i = nums.length - 2;
		while (i >= 0 && nums[i + 1] <= nums[i])
			i--;
		if (i >= 0) {
			int j = nums.length - 1;
			while (j >= 0 && nums[j] <= nums[i])
				j--;
			swap(nums, i, j);
		}
		reverse(nums, i + 1, nums.length - 1);
	}

	private static void swap(int nums[], int i, int j) {
		int temp = nums[i];
		nums[i] = nums[j];
		nums[j] = temp;
	}

	private static void reverse(int[] nums, int i, int j) {
		while (i < j)
			swap(nums, i++, j--);
	}
	
	// 32.Longest Valid Parentheses
	// see stack is empty or not, if current char is left bracket we add it,else we
	// pop out.
	public static int longestValidParentheses(String s) {
		int length = 0;
		if (s.length() == 0)
			return 0;

		Stack<Integer> st = new Stack<>();
		st.push(-1);
		for (int i = 0; i < s.length(); i++) {
			int top = st.peek();
			if (top != -1 && s.charAt(i) == ')' && s.charAt(top) == '(') {
				st.pop();
				length = Math.max(length, i - st.peek());
			} else {
				st.add(i);
			}
		}
		return length;
	}

	// 33.Search in Rotated Sorted Array
	// [4,5,6,7,0,1,2]
	// [0,1,2,4,5,6,7]
	public static int search(int[] nums, int target) {
		int left = 0;
		int right = nums.length - 1;
		while (right >= left) {
			int mid = left + (right - left) / 2;
			if (nums[mid] == target)
				return mid;
			if (nums[mid] < nums[right]) {
				if (target > nums[mid] && nums[right] >= target)
					left = mid + 1;
				else
					right = mid - 1;

			} else {
				if (target < nums[mid] && nums[left] <= target)
					right = mid - 1;
				else
					left = mid + 1;

			}
		}
		return -1;
	}

	public static int gcd(int x, int y) {
		int tmp;
		while (x % y != 0) {
			tmp = y;
			y = x % y;
			x = tmp;
		}
		return y;
	}

	// 34. Find First and Last Position of Element in Sorted Array
	public static int[] searchRange(int[] nums, int target) {
		var fIndex = findStartIndex(nums, target);
		var lIdex = findLastIndex(nums, target);
		int result[] = new int[2];
		result[0] = fIndex;
		result[1] = lIdex;
		return result;
	}

	private static int findStartIndex(int nums[], int target) {
		int left = 0;
		int right = nums.length - 1;
		int index = -1;
		while (right >= left) {
			int mid = left + right >> 1;
			if (nums[mid] >= target)
				right = mid - 1;
			else if (nums[mid] <= target)
				left = mid + 1;
			if (nums[mid] == target)
				index = mid;

		}
		return index;
	}

	private static int findLastIndex(int nums[], int target) {
		int left = 0;
		int right = nums.length - 1;
		int index = -1;
		while (left <= right) {
			int mid = left + right >> 1;
			if (nums[mid] <= target)
				left = mid + 1;
			else if (nums[mid] >= target)
				right = mid - 1;
			if (nums[mid] == target)
				index = mid;

		}
		return index;
	}

	// 35.Search Insert Position
	// Binary Search
	public static int searchInsert(int[] nums, int target) {
//		O(n)
//		for (int i = 0; i < nums.length; ++i) {
//            if (nums[i] >= target) return i;
//        }
//        return nums.length;
//		O(logN)

		int left = 0, right = nums.length - 1;
		while (left + 1 < right) {
			int mid = left + (right - left) / 2;
			if (nums[mid] == target)
				return mid;
			else if (nums[mid] > target)
				right = mid;
			else
				left = mid;

		}
		// 當target 比所有數字小的時候
		if (nums[left] >= target && nums[right] >= target)
			return left;
		// target 沒出現在陣列但在其範圍內
		else if (nums[left] <= target && nums[right] >= target)
			return right;
		// target 大於陣列中所有數字
		else
			return right + 1;

	}

	// 36.Valid Sudoku
	// count 9 block, and 3 rows 3 columns in the every block
	public static boolean isValidSudoku(char[][] board) {
		var result = true;
		for (int i = 0; i < 9; i++) {
			result = checkRowValid(board, i);
			if (!result)
				return result;

		}

		for (int i = 0; i < 9; i++) {
			result = checkColValid(board, i);
			if (!result)
				return result;

		}

		for (int i = 0; i < 9; i += 3) {
			for (int j = 0; j < 9; j += 3) {
				result = checkValidBlock(board, i, j);
				if (!result)
					return result;
			}
		}
		return true;
	}

	private static boolean checkRowValid(char[][] board, int row) {
		HashSet<Character> repeat = new HashSet<>();
		for (int i = 0; i < 9; i++) {
			if (board[row][i] != '.') {
				if (repeat.contains(board[row][i]))
					return false;

				repeat.add(board[row][i]);
			}
		}
		return true;
	}

	private static boolean checkColValid(char[][] board, int col) {
		HashSet<Character> repeat = new HashSet<>();
		for (int i = 0; i < 9; i++) {
			if (board[i][col] != '.') {
				if (repeat.contains(board[i][col]))
					return false;

				repeat.add(board[i][col]);
			}
		}
		return true;
	}

	private static boolean checkValidBlock(char[][] board, int leftTopRow, int leftTopCol) {
		HashSet<Character> repeat = new HashSet<>();
		for (int i = leftTopRow; i < leftTopRow + 3; i++) {
			for (int j = leftTopCol; j < leftTopCol + 3; j++) {
				if (board[i][j] != '.')
					if (repeat.contains(board[i][j]))
						return false;
				repeat.add(board[i][j]);

			}
		}
		return true;

	}

	// 37.Sudoku Solver
	// need isVaild() to judge is Validate Sudoku, use recursion to try all the
	// result step by step.
	public static void solveSudoku(char[][] board) {
		traverse(board, 0, 0);
	}

	private static boolean traverse(char[][] board, int i, int j) {
		// we traverse each cell from left to right, when the column is over 9,
		// we move to next row, and when the row is 9 means it's in the end.
		if (i == 9)
			return true;
		if (j >= 9)
			return traverse(board, i + 1, 0);
		if (board[i][j] != '.')
			return traverse(board, i, j + 1);
		// try numbers of 1-9
		for (int c = '1'; c <= '9'; c++) {
			if (!isValid(board, i, j, c))
				continue;
			board[i][j] = (char) c;
			if (traverse(board, i, j + 1))
				return true;
			board[i][j] = '.';

		}
		return false;
	}

	// check if the board is valid with the constraint.
	private static boolean isValid(char board[][], int i, int j, int c) {
		for (int x = 0; x < 9; x++) {
			if (board[x][j] == c)
				return false;
		}

		for (int y = 0; y < 9; y++) {
			if (board[i][y] == c) {
				return false;
			}
		}
		// block index
		int row = i - i % 3;
		int col = j - j % 3;
		for (int x = 0; x < 3; x++) {
			for (int y = 0; y < 3; y++) {
				if (board[row + x][col + y] == c)
					return false;
			}
		}
		return true;
	}

	// 38.Count and Say
	public static String countAndSay(int n) {
		int i = 1;
		String result = "1";
		if (n == 1)
			return result;

		while (n > i) {
			int count = 1;
			StringBuilder sb = new StringBuilder();
			for (int j = 1; j < result.length(); j++) {
				if (result.charAt(j) == result.charAt(j - 1)) {
					count++;
				} else {
					sb.append(count).append(result.charAt(j - 1));
					count = 1;
				}
			}
			sb.append(count).append(result.charAt(result.length() - 1));
			result = sb.toString();
			i++;
		}
		return result;
	}

	// 39. Combination Sum
	// DFS concept
	public static List<List<Integer>> combinationSum(int[] candidates, int target) {
		List<List<Integer>> res = new ArrayList<>();
		dfs(0, target, new ArrayList<Integer>(), candidates, res);
		return res;

	}

	private static void dfs(int start, int target, ArrayList<Integer> list, int[] candidates, List<List<Integer>> res) {
		if (target == 0) {
			res.add(new ArrayList<>(list));
			return;
		}

		if (target < 0)
			return;

		for (int i = start; i < candidates.length; i++) {
			list.add(candidates[i]);
			dfs(i, target - candidates[i], list, candidates, res);
			list.remove(list.size() - 1);
		}

	}

	// 40.CombinationSum2
	// dps concept ,but we need to sort and avoid using same element.
	public static List<List<Integer>> combinationSum2(int[] candidates, int target) {
		List<List<Integer>> res = new ArrayList<>();
		Arrays.sort(candidates);
		helper(0, target, res, new ArrayList<Integer>(), candidates);
		return res;
	}

	private static void helper(int start, int target, List<List<Integer>> res, ArrayList<Integer> list,
			int[] candidates) {
		if (target == 0) {
			res.add(new ArrayList<>(list));
			return;
		}
		if (target < 0)
			return;

		for (int i = start; i < candidates.length; i++) {
			if (start < i && candidates[i] == candidates[i - 1])
				continue;
			list.add(candidates[i]);
			helper(i + 1, target - candidates[i], res, list, candidates);
			list.remove(list.size() - 1);
		}
	}

	// 268.missingNumbers
//	public static int missingNumber(int[] nums) {
//		int sum = 0;
//		for (int i = 0; i < nums.length; i++) {
//			sum += nums[i];
//		}
//		return (nums.length) * (nums.length + 1) / 2 - sum;
//	}

	public static int missingNumber(int[] nums) {
		Set<Integer> numSet = new HashSet<Integer>();
		for (int num : nums)
			numSet.add(num);

		int expectedNumCount = nums.length + 1;
		for (int number = 0; number < expectedNumCount; number++) {
			if (!numSet.contains(number)) {
				return number;
			}
		}
		return -1;
	}
	
	// 41.First missing positive
	// use indexing sort
	public static int firstMissingPositive(int[] nums) {
		if (nums.length == 0 || nums == null)
			return 1;

		for (int i = 0; i < nums.length; i++) {
			// 需是正數且小於自身長度，及避免相鄰兩數相同造成死循環
			if (nums[i] <= nums.length && nums[i] > 0 && nums[nums[i] - 1] != nums[i]) {
				int temp = nums[nums[i] - 1];
				nums[nums[i] - 1] = nums[i];
				nums[i] = temp;
				// 始終在第一個數的地方交換
				i--;
			}
		}

		for (int i = 0; i < nums.length; i++)
			if (nums[i] != i + 1)
				return i + 1;

		return nums.length + 1;
	}

	// 43.Multiply Strings
	public static String multiply(String num1, String num2) {
		String n1 = new StringBuilder(num1).reverse().toString();
		String n2 = new StringBuilder(num2).reverse().toString();

		int[] d = new int[num1.length() + num2.length()];

		// multiply each digit and sum at the corresponding positions
		for (int i = 0; i < n1.length(); i++) {
			for (int j = 0; j < n2.length(); j++) {
				d[i + j] += (n1.charAt(i) - '0') * (n2.charAt(j) - '0');

			}
		}
		StringBuilder sb = new StringBuilder();

		// calculate each digit
		for (int i = 0; i < d.length; i++) {
			int mod = d[i] % 10;
			int carry = d[i] / 10;
			if (i + 1 < d.length)
				d[i + 1] += carry;

			sb.insert(0, mod);
		}

		// remove the number of 0 in index[0]
		while (sb.charAt(0) == '0' && sb.length() > 1)
			sb.deleteCharAt(0);

		return sb.toString();

	}
}
