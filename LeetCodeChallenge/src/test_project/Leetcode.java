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
}
