//date: 2023-02-13T17:04:54Z
//url: https://api.github.com/gists/f7179ca2a62072c7d1987f882f513112
//owner: https://api.github.com/users/Sam-Belliveau

import java.util.LinkedList;
import java.util.Optional;
import java.util.Queue;

public class ArmBFSField {

    public static final int kDegreeRange = 360;

    // This value does not matter as much as long as its small.
    public static final double kConstraintDistCost = 0.1;
    public static final double kArmCostPerDeg = 1.0;
    public static final double kWristCostPerDeg = 1.0;

    public static final int kArmNodeSpeed = 1;
    public static final int kWristNodeSpeed = 2;

    private static int normalize(int degrees) {
        while (degrees < 0) {
            degrees += kDegreeRange;
        }

        while (kDegreeRange <= degrees) {
            degrees -= kDegreeRange;
        }

        return degrees;
    }

    private static int normalizeDistance(int degrees) {
        while (degrees < -kDegreeRange / 2) {
            degrees += kDegreeRange;
        }

        while (kDegreeRange / 2 <= degrees) {
            degrees -= kDegreeRange;
        }

        return degrees;
    }

    public interface Constraint {
        public boolean isInvalid(int armDeg, int wristDeg);

        public default Constraint add(Constraint next) {
            return (a, w) -> this.isInvalid(a, w) || next.isInvalid(a, w);
        }
    }

    public class Node {

        private final boolean mValid;

        private final int mArmDeg;
        private final int mWristDeg;

        private double mConstraintCost;
        private double mSetpointCost;

        private Node mNextNode;

        public Node(int armDeg, int wristDeg, boolean valid) {
            mValid = valid;

            mArmDeg = normalize(armDeg);
            mWristDeg = normalize(wristDeg);

            mConstraintCost = valid ? Double.MAX_VALUE / 4.0 : 0.0;
            mSetpointCost = Double.MAX_VALUE / 2.0;

            mNextNode = this;
        }

        private double getDistanceCost(Node previous) {
            final double dx = normalizeDistance(this.mArmDeg - previous.mArmDeg);
            final double dy = normalizeDistance(this.mWristDeg - previous.mWristDeg);
            return Math.hypot(
                    Math.abs(dx) * kArmCostPerDeg,
                    Math.abs(dy) * kWristCostPerDeg);
        }

        private Node makeSetpoint() {
            if (!mValid)
                throw new RuntimeException("Setpoint is within constraint!");

            mNextNode = this;
            mSetpointCost = 0;
            return this;
        }

        private Optional<Node> expandConstraint(Node previous) {
            if (this.equals(previous)) {
                return Optional.empty();
            }

            final double cost = this.getDistanceCost(previous) * kConstraintDistCost;
            final double newConstraintCost = previous.mConstraintCost + cost;

            if (newConstraintCost < mConstraintCost) {
                mConstraintCost = newConstraintCost;
                return Optional.of(this);
            }

            return Optional.empty();
        }

        private Optional<Node> expandSearch(Node previous) {
            // do not expand searches into invalid areas
            if (this.equals(previous) || !mValid || !previous.mValid) {
                return Optional.empty();
            }

            // the difference in constraint cost between adjacent nodes is small
            // so this will basically find the shortest path, but if there is an
            // equidistant path that is further from the constraints, take that one instead,
            final double cost = this.getDistanceCost(previous);
            final double newSetpointCost = previous.mSetpointCost + cost;
            final double newCost = newSetpointCost - previous.mConstraintCost;
            final double oldCost = mSetpointCost - mNextNode.mConstraintCost;

            if (newCost < oldCost) {
                mSetpointCost = newSetpointCost;
                mNextNode = previous;

                return Optional.of(this);
            }

            return Optional.empty();
        }

        private Optional<Node> expandEscape(Node previous) {
            // a valid node should not escape as it is valid
            if (this.equals(previous) || mValid) {
                return Optional.empty();
            }

            // when escaping you only want to move one mech at a time
            if (mArmDeg != previous.mArmDeg && mWristDeg != previous.mWristDeg) {
                return Optional.empty();
            }

            // if there is a new shorter escape path, take it
            final double cost = this.getDistanceCost(previous);
            final double newSetpointCost = previous.mSetpointCost + cost;

            if (newSetpointCost < mSetpointCost) {
                mSetpointCost = newSetpointCost;
                mNextNode = previous;

                return Optional.of(this);
            }

            return Optional.empty();
        }

        public boolean isValid() {
            return mValid;
        }

        public int getArmDeg() {
            return mArmDeg;
        }

        public int getWristDeg() {
            return mWristDeg;
        }

        public Node next() {
            return mNextNode;
        }

        public Node next(int depth) {
            Node result = this;

            while (depth-- > 0) {
                result = result.next();
            }

            return result;
        }
    }

    private Node[][] mNodeMap;

    public ArmBFSField(int targetArmDeg, int targetWristDeg, Constraint constraints) {
        mNodeMap = new Node[kDegreeRange][kDegreeRange];

        Queue<Node> constraintOpenSet = new LinkedList<>();

        // initialize all nodes and test to see if they fit the constraints
        for (int arm = 0; arm < kDegreeRange; ++arm) {
            for (int wrist = 0; wrist < kDegreeRange; ++wrist) {
                final Node node = new Node(arm, wrist, !constraints.isInvalid(arm, wrist));

                // if the node is a member of the constraints,
                // add it to the constraint openset for later
                if (!node.isValid()) {
                    constraintOpenSet.add(node);
                }

                setNode(arm, wrist, node);
            }
        }

        // we want to be able to find the shortest path that also remains the
        // furthest from all the constraints. doing a search will let us determine
        // the distance each node is from the closest constraint.

        Node next;
        while ((next = constraintOpenSet.poll()) != null) {
            final int arm = next.getArmDeg();
            final int wrist = next.getWristDeg();

            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    final Node node = getNode(arm + dx, wrist + dy);
                    final Optional<Node> dir = node.expandConstraint(next);

                    if (dir.isPresent()) {
                        constraintOpenSet.add(dir.get());
                    }
                }
            }
        }

        // do a standard BFS on all of the valid nodes.
        // record the edges of the constraints for later.

        Queue<Node> searchOpenSet = new LinkedList<>();
        Queue<Node> escapeOpenSet = new LinkedList<>();

        searchOpenSet.add(getNode(targetArmDeg, targetWristDeg).makeSetpoint());

        while ((next = searchOpenSet.poll()) != null) {
            final int arm = next.getArmDeg();
            final int wrist = next.getWristDeg();

            for (int dx = -kArmNodeSpeed; dx <= kArmNodeSpeed; ++dx) {
                for (int dy = -kWristNodeSpeed; dy <= kWristNodeSpeed; ++dy) {
                    final Node node = getNode(arm + dx, wrist + dy);
                    final Optional<Node> search = node.expandSearch(next);

                    if (search.isPresent()) {
                        searchOpenSet.add(search.get());
                    } else {
                        // if a node is invalid here, it means that it is touching
                        // a valid node, so it will be the beginning of our escape path.
                        final Optional<Node> escape = node.expandEscape(next);

                        if (escape.isPresent()) {
                            escapeOpenSet.add(escape.get());
                        }
                    }
                }
            }
        }

        // do a BFS on all of the invalid nodes so that
        // they can lead their way to the closes setpoint.

        while ((next = escapeOpenSet.poll()) != null) {
            final int arm = next.getArmDeg();
            final int wrist = next.getWristDeg();

            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    final Node node = getNode(arm + dx, wrist + dy);
                    final Optional<Node> escape = node.expandEscape(next);

                    if (escape.isPresent()) {
                        escapeOpenSet.add(escape.get());
                    }
                }
            }
        }
    }

    private void setNode(int armDeg, int wristDeg, Node node) {
        mNodeMap[normalize(armDeg)][normalize(wristDeg)] = node;
    }

    public Node getNode(int armDeg, int wristDeg) {
        return mNodeMap[normalize(armDeg)][normalize(wristDeg)];
    }

    public static void main(String... args) {

        Constraint constraints = (a, w) -> a <= 180;
        constraints = constraints.add(
                (a, w) -> (Math.abs(a - 270) < 30) && ((w < 80) || (w > 100)));

        ArmBFSField field = new ArmBFSField(200, 270, constraints);

        Node node = field.getNode(330, 0).next();

        System.out.println("arm,wrist");
        while (node != node.next()) {
            System.out.println(node.getArmDeg() + "," + node.getWristDeg());
            node = node.next(3);
        }
    }
}