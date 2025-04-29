use petgraph::graph::{IndexType, NodeIndex};
use petgraph::{Graph, Undirected};
use std::collections::{BTreeMap, BTreeSet};

/// Uses the [Louvain method][louvain] to find communities.
///
/// Returns a [dendrogram][dendrogram] containing the communities of each iteration.
/// At least the initial state is included where each node is in its own community.
///
/// The subsequent iterations cluster the supergraph of the previous iteration until no more
/// modularity optimization is possible.
///
/// [louvain]: https://en.wikipedia.org/wiki/Louvain_method
/// [dendrogram]: https://en.wikipedia.org/wiki/Dendrogram
pub fn louvain<N, E, Ix>(graph: &Graph<N, E, Undirected, Ix>) -> Vec<Vec<Vec<NodeIndex<Ix>>>>
where
    N: Default,
    E: Copy + Into<f64> + std::iter::Sum,
    Ix: IndexType,
{
    let mut louvain = Louvain::new(graph);
    let mut dendrogram: Vec<Vec<Vec<NodeIndex<Ix>>>> = vec![louvain.communities()];

    // The graph of the communities from the previous iteration.
    let mut supergraph;

    // Run until no more optimization is possible.
    while louvain.optimize() {
        // The communities of a supergraph need to be replaced by the original node indices.
        let communities = louvain
            .communities()
            .into_iter()
            .map(|community| {
                // Map each node to the nodes its community is representing.
                community
                    .into_iter()
                    .flat_map(|node| dendrogram.last().unwrap()[node.index()].iter().copied())
                    .collect()
            })
            .collect();

        // Add the communities of the current iteration to the dendrogram.
        dendrogram.push(communities);

        // Aggregate the graph for the next iteration.
        supergraph = louvain.aggregate();

        // Prepare the next iteration.
        louvain = Louvain::new(&supergraph)
    }

    dendrogram
}

/// Data structure to control the process of the louvain method.
#[derive(Debug)]
struct Louvain<'a, N, E, Ix>
where
    Ix: IndexType,
{
    /// The graph the algorithm operates on.
    graph: &'a Graph<N, E, Undirected, Ix>,
    /// Nodes in each community.
    communities: Vec<BTreeSet<NodeIndex<Ix>>>,
    /// Community memberships of each node.
    memberships: BTreeMap<NodeIndex<Ix>, usize>,
    /// `2m`: Total (edge) weight of the graph `* 2` (covering both directions).
    two_m: f64,
    /// `k_i`: Total edge weight of each node.
    node_weights: Vec<f64>,
    /// `Σ_tot`: Total weight of each community.
    sigma_tot: Vec<f64>,
}

impl<'a, N, E, Ix> Louvain<'a, N, E, Ix>
where
    N: Default,
    E: Copy + Into<f64> + std::iter::Sum,
    Ix: IndexType,
{
    pub fn new(graph: &'a Graph<N, E, Undirected, Ix>) -> Self {
        // Initially, each node forms its own community.
        let communities = graph
            .node_indices()
            .map(|node| BTreeSet::from([node]))
            .collect();

        let memberships = graph
            .node_indices()
            .enumerate()
            .map(|(community, node)| (node, community))
            .collect();

        // Weight of all edges.
        let m: E = graph.edge_weights().copied().sum();

        // Weight of each edge incident to a node.
        let node_weights: Vec<f64> = graph
            // Consider each node.
            .node_indices()
            .map(|node| {
                // Get all neighbors of a node ...
                graph
                    .neighbors(node)
                    // and sum the weight of their link.
                    .map(|neighbor| {
                        *graph
                            .edge_weight(graph.find_edge(node, neighbor).unwrap())
                            .unwrap()
                    })
                    .sum::<E>()
                    .into()
            })
            .collect();

        Self {
            graph,
            communities,
            memberships,
            two_m: m.into() * 2f64,
            // As each node forms its own community, the total weight of these communities is the
            // respective node weight.
            sigma_tot: node_weights.clone(),
            node_weights,
        }
    }

    /// Returns the (current) communities / partition.
    pub fn communities(&self) -> Vec<Vec<NodeIndex<Ix>>> {
        self.communities
            .iter()
            .map(BTreeSet::iter)
            .map(Iterator::copied)
            .map(Vec::from_iter)
            .filter(|community| !community.is_empty())
            .collect()
    }

    /// Phase 1: Modularity Optimization
    ///
    /// Assigns each node to the best community until no more changes occur,
    /// returning whether any optimization took place.
    /// Communities can be extracted via [communities][Self::communities].
    fn optimize(&mut self) -> bool {
        // Take a copy of the nodes to iterate over them.
        let nodes: Vec<NodeIndex<Ix>> = self.graph.node_indices().collect();

        // Flag for tracking whether any optimization took place.
        let mut optimized = false;

        // Flag for tracking whether another run is needed.
        let mut run_again = true;

        // Move nodes as long as updates keep occurring.
        while run_again {
            run_again = false;
            nodes.iter().for_each(|&node| {
                if self.move_node(node) {
                    run_again = true;
                    optimized = true;
                }
            });
        }

        optimized
    }

    /// Phase 2: Community Aggregation
    ///
    /// Creates a new graph in which each community is represented by a single node.
    fn aggregate(&self) -> Graph<N, E, Undirected, Ix> {
        // Create a copy of the communities to iterate over them.
        let communities = self.communities();

        // Build the edges by iterating over the communities.
        let edges = communities.iter().enumerate().flat_map(|(i, community)| {
            // Combine each community with each other community.
            communities
                .iter()
                .enumerate()
                // Consider each combination only once. This includes the combination of each
                // community with itself.
                .skip(i)
                .filter_map(move |(j, other)| {
                    // Each community pair is mapped to the edges connecting them, if any.
                    let mut edges = community
                        .iter()
                        // Check for each node of one community ...
                        .flat_map(|&node_of_community| {
                            other
                                .iter()
                                // ... whether it has an edge to the other community.
                                .filter_map(move |&node_of_other| {
                                    self.graph.find_edge(node_of_community, node_of_other)
                                })
                                // If so, take the edge weight.
                                .map(|edge| self.graph.edge_weight(edge).unwrap())
                        })
                        .copied()
                        .peekable();

                    // Check if there are any edges. If not, return nothing.
                    edges.peek()?;

                    // Create an edge connecting the communities with the weight being the sum
                    // of all connecting edges.
                    Some((Ix::new(i), Ix::new(j), edges.sum::<E>()))
                })
        });

        Graph::from_edges(edges)
    }

    /// `k_i,in`: Weight of edges from a node into a community.
    ///
    /// Self edges are not considered.
    fn k_i_in(&self, node: NodeIndex<Ix>, community: usize) -> f64 {
        self.communities[community]
            .iter()
            .filter(|&&other| other != node)
            .filter_map(|&other| self.graph.find_edge(node, other))
            .map(|edge| self.graph.edge_weight(edge).unwrap())
            .copied()
            .sum::<E>()
            .into()
    }

    /// Modularity change by moving a node into a community
    ///
    /// The node is expected to **not** be in this community.
    ///
    /// # Note
    ///
    /// The delta modularity function is the same as in the [official implementation][sourceforge]:
    ///
    /// `k_i_n - Σ_tot * k_i / 2m`
    ///
    /// This is different to the function used in the [original paper][paper], which produces
    /// different results. See [here][discussion] for a discussion.
    ///
    /// [sourceforge]: https://sourceforge.net/projects/louvain
    /// [paper]: https://perso.uclouvain.be/vincent.blondel/publications/08BG.pdf
    /// [discussion]: https://mathoverflow.net/questions/414575
    fn delta(&self, node: NodeIndex<Ix>, community: usize) -> f64 {
        // Sum of weights of edges to nodes of the community.
        let sigma_tot = self.sigma_tot[community];

        // Edge weight of the node.
        let k_i = self.node_weights[node.index()];

        // Weight of edges from the node into the community.
        let k_i_in = self.k_i_in(node, community);

        k_i_in - sigma_tot * k_i / self.two_m
    }

    /// Returns the set of communities neighboring a node.
    fn incident_communities(&self, node: NodeIndex<Ix>) -> BTreeSet<usize> {
        self.graph
            .neighbors(node)
            .filter(|&neighbor| neighbor != node)
            .map(|neighbor| self.memberships[&neighbor])
            .collect()
    }

    /// Returns the best community to put a node in when there is one.
    ///
    /// When no better options than the current community is found, `None` is returned.
    fn best_community(&self, node: NodeIndex<Ix>) -> Option<usize> {
        // Find the best community and its modularity change.
        let (best, delta) = self
            // Consider each neighboring community.
            .incident_communities(node)
            .iter()
            // Calculate the modularity change for moving into the community.
            .map(|&community| (community, self.delta(node, community)))
            // Take the biggest change.
            .max_by(|(_, delta_a), (_, delta_b)| delta_a.total_cmp(delta_b))
            // Default to a negative value when nothing was found to stay in the current community.
            .unwrap_or((0, -1f64));

        // In case the best change is negative, the node remains in its current community.
        if delta <= 0f64 {
            return None;
        }

        Some(best)
    }

    /// Removes a node from its current community.
    fn remove(&mut self, node: NodeIndex<Ix>) {
        // Get the community the node was in and remove the community membership.
        let community = self
            .memberships
            .remove(&node)
            .expect("Node to be removed is not in a community.");

        // Remove the node from the community.
        assert!(
            self.communities[community].remove(&node),
            "Node to be removed should be part of a community."
        );

        // Decrease the total weight of the community by the nodes weight.
        self.sigma_tot[community] -= self.node_weights[node.index()];
    }

    /// Inserts a node into the given community.
    fn insert(&mut self, node: NodeIndex<Ix>, community: usize) {
        // Enter the community membership.
        assert!(
            self.memberships.insert(node, community).is_none(),
            "Node to be inserted should not have been in a community before."
        );

        // Insert the node into the community.
        assert!(
            self.communities[community].insert(node),
            "Node to be inserted should not have been in a community before."
        );

        // Increase the total weight of the community by the nodes weight.
        self.sigma_tot[community] += self.node_weights[node.index()];
    }

    /// Moves a node into the best community, returning whether a change actually occurred.
    ///
    /// Removes the node, calculates the best community and places the node there.
    fn move_node(&mut self, node: NodeIndex<Ix>) -> bool {
        // Note the current community and then remove the node.
        let current = self.memberships[&node];
        self.remove(node);

        // Find the best community to put the node in.
        let new = if let Some(best) = self.best_community(node) {
            best
        } else {
            current
        };

        // Inset it there.
        self.insert(node, new);

        // Return whether this was actually a change or whether the node simply stayed where it was.
        current != new
    }
}

#[cfg(test)]
mod test {
    use super::louvain;
    use petgraph::graph::NodeIndex;
    use petgraph::{Graph, Undirected};

    /// Generates the example graph from the [Louvain paper][paper].
    ///
    /// [paper]: https://perso.uclouvain.be/vincent.blondel/publications/08BG.pdf
    fn graph() -> Graph<u8, u8, Undirected> {
        let mut graph = Graph::<u8, u8, Undirected>::new_undirected();

        for i in 0..=15 {
            graph.add_node(i);
        }

        graph.extend_with_edges(
            [
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 2),
                (1, 4),
                (1, 7),
                (2, 4),
                (2, 5),
                (2, 6),
                (3, 7),
                (4, 10),
                (5, 7),
                (5, 11),
                (6, 7),
                (6, 11),
                (8, 9),
                (8, 10),
                (8, 11),
                (8, 14),
                (8, 15),
                (9, 12),
                (9, 14),
                (10, 11),
                (10, 12),
                (10, 13),
                (10, 14),
                (11, 13),
            ]
            .iter()
            .map(|(a, b)| (*a, *b, 1))
            .collect::<Vec<_>>(),
        );

        graph
    }

    #[test]
    fn clustering() {
        let graph = graph();
        assert_eq!(
            louvain(&graph),
            vec![
                vec![
                    vec![NodeIndex::new(0),],
                    vec![NodeIndex::new(1),],
                    vec![NodeIndex::new(2),],
                    vec![NodeIndex::new(3),],
                    vec![NodeIndex::new(4),],
                    vec![NodeIndex::new(5),],
                    vec![NodeIndex::new(6),],
                    vec![NodeIndex::new(7),],
                    vec![NodeIndex::new(8),],
                    vec![NodeIndex::new(9),],
                    vec![NodeIndex::new(10),],
                    vec![NodeIndex::new(11),],
                    vec![NodeIndex::new(12),],
                    vec![NodeIndex::new(13),],
                    vec![NodeIndex::new(14),],
                    vec![NodeIndex::new(15),],
                ],
                vec![
                    vec![NodeIndex::new(3), NodeIndex::new(6), NodeIndex::new(7),],
                    vec![
                        NodeIndex::new(0),
                        NodeIndex::new(1),
                        NodeIndex::new(2),
                        NodeIndex::new(4),
                        NodeIndex::new(5),
                    ],
                    vec![
                        NodeIndex::new(8),
                        NodeIndex::new(9),
                        NodeIndex::new(10),
                        NodeIndex::new(12),
                        NodeIndex::new(14),
                        NodeIndex::new(15),
                    ],
                    vec![NodeIndex::new(11), NodeIndex::new(13),],
                ],
                vec![
                    vec![
                        NodeIndex::new(3),
                        NodeIndex::new(6),
                        NodeIndex::new(7),
                        NodeIndex::new(0),
                        NodeIndex::new(1),
                        NodeIndex::new(2),
                        NodeIndex::new(4),
                        NodeIndex::new(5),
                    ],
                    vec![
                        NodeIndex::new(8),
                        NodeIndex::new(9),
                        NodeIndex::new(10),
                        NodeIndex::new(12),
                        NodeIndex::new(14),
                        NodeIndex::new(15),
                        NodeIndex::new(11),
                        NodeIndex::new(13),
                    ],
                ],
            ]
        );
    }
}
