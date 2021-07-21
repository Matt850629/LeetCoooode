package test_project;

public class ListNode {
	int value;
	ListNode next;

	public ListNode(int value) {
		this.value = value;
	}

	ListNode(int value, ListNode next) {
		this.value = value;
		this.next = next;
	}

	@Override
	public String toString() {
		return "nodeVal: " + value;
	}

}
