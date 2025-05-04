use petgraph::adj::IndexType;
use petgraph::csr::Csr;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeIdentifiers};
use petgraph::Undirected;
use std::collections::BTreeMap;
use std::ops::{Add, AddAssign};

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
pub fn louvain<N, E, Ix>(graph: &Csr<N, E, Undirected, Ix>) -> Vec<Vec<Vec<Ix>>>
where
    E: Copy + Ord + Default + Into<f64> + Add<Output = E> + AddAssign + std::iter::Sum<E>,
    N: Default,
    Ix: IndexType,
{
    let mut louvain = Louvain::new(graph);
    let mut dendrogram: Vec<Vec<Vec<Ix>>> = vec![louvain.communities()];

    // The graph of the communities from the previous iteration.
    let mut supergraph;

    // Run until no more optimization is possible.
    while louvain.optimize() {
        // Replace the supergraph communities by the original node indices.
        let communities = louvain
            .communities()
            .into_iter()
            .map(|community| {
                // Map each supernode to the nodes its community is representing.
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
        louvain = Louvain::new(&supergraph);
    }

    dendrogram
}

/// Data structure to control the process of the louvain method.
struct Louvain<'a, N, E, Ix> {
    /// The graph the algorithm operates on.
    graph: &'a Csr<N, E, Undirected, Ix>,
    /// Community memberships of each node.
    memberships: Vec<Option<usize>>,
    /// `2m`: Total (edge) weight of the graph `* 2` (covering both directions).
    two_m: f64,
    /// `k_i`: Total edge weight of each node.
    node_weights: Vec<f64>,
    /// `Σ_tot`: Total weight of each community.
    sigma_tot: Vec<f64>,
}

impl<'a, N, E, Ix> Louvain<'a, N, E, Ix>
where
    E: Copy + Ord + Default + Into<f64> + Add<Output = E> + AddAssign + std::iter::Sum<E>,
    N: Default,
    Ix: IndexType,
{
    pub fn new(graph: &'a Csr<N, E, Undirected, Ix>) -> Self {
        // Initially, each node forms its own community.
        let memberships = (0..graph.node_count()).map(Some).collect();

        // Sum the weight of all edges two times. Self-edges are taken two times, while others
        // are occurring in both directions and therefore only taken once per direction.
        let two_m: f64 = graph
            .edge_references()
            .map(|edge| {
                if edge.source() == edge.target() {
                    let weight = *edge.weight();
                    weight + weight
                } else {
                    *edge.weight()
                }
            })
            .sum::<E>()
            .into();

        // Edge weight of each node.
        let node_weights: Vec<f64> = graph
            // Consider each node.
            .node_identifiers()
            .map(|node| {
                // Add the weights of all edges.
                graph
                    .edges(node)
                    .map(|edge| edge.weight())
                    .copied()
                    .sum::<E>()
                    .into()
            })
            .collect();

        Self {
            graph,
            memberships,
            two_m,
            // As each node forms its own community, the total weight of these communities is the
            // respective node weight.
            sigma_tot: node_weights.clone(),
            node_weights,
        }
    }

    /// Returns the (current) communities.
    pub fn communities(&self) -> Vec<Vec<Ix>> {
        let mut communities = vec![Vec::new(); self.graph.node_count()];

        // For each node, enter its community membership in the vector.
        self.graph.node_identifiers().for_each(|node| {
            let community =
                self.memberships[node.index()].expect("Node should be part of a community.");

            communities[community].push(node);
        });

        // Only keep communities with nodes in them.
        communities.retain(|community| !community.is_empty());
        communities
    }

    /// Normalizes the community memberships of nodes so that the community indices represent a contiguous range.
    fn normalize_memberships(&mut self) {
        // `communities()` only returns non-empty communities.
        self.communities()
            .into_iter()
            .enumerate()
            // In each community, reset the membership to the index of the contiguous range.
            .for_each(|(i, community)| {
                community.into_iter().for_each(|node| {
                    self.memberships[node.index()] = Some(i);
                });
            });
    }

    /// Phase 1: Modularity Optimization
    ///
    /// Assigns each node to the best community until no more changes occur,
    /// returning whether any optimization took place.
    /// Communities can be extracted via [communities][Self::communities].
    fn optimize(&mut self) -> bool {
        // Take a copy of the nodes to iterate over them.
        let nodes: Vec<Ix> = self.graph.node_identifiers().collect();

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

        self.normalize_memberships();
        optimized
    }

    /// Phase 2: Community Aggregation
    ///
    /// Creates a new graph in which each community is represented by a single node.
    fn aggregate(&self) -> Csr<N, E, Undirected, Ix> {
        // Map for adding all individual edges.
        let mut edges: BTreeMap<(Ix, Ix), E> = BTreeMap::new();

        // Add each edge while using the community membership as source/target.
        self.graph.edge_references().for_each(|edge| {
            let a = Ix::new(self.memberships[edge.source().index()].unwrap());
            let b = Ix::new(self.memberships[edge.target().index()].unwrap());
            let weight = *edge.weight();
            *edges.entry((a, b)).or_default() += weight;
        });

        // Convert the map to a vector.
        // Because the edges are built using a `BTreeMap`, they are already sorted.
        let edges = edges
            .into_iter()
            .map(|((a, b), weight)| (a, b, weight))
            .collect::<Vec<_>>();

        Csr::from_sorted_edges(&edges).expect("Failed to build supergraph.")
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
    fn delta(&self, node: Ix, community: usize, k_i_in: E) -> f64 {
        // Sum of weights of edges to nodes of the community.
        let sigma_tot = self.sigma_tot[community];

        // Edge weight of the node.
        let k_i = self.node_weights[node.index()];

        // Weight of edges from node to community.
        let k_i_in = k_i_in.into();

        k_i_in - sigma_tot * k_i / self.two_m
    }

    /// Returns the set of communities neighboring a node together with the respective `k_i_in` values.
    fn incident_communities(&self, node: Ix) -> BTreeMap<usize, E> {
        let mut communities: BTreeMap<usize, E> = BTreeMap::new();

        // For each edge from the node, note the community of the connected node and add the edge
        // weight to the communities `k_i_in` value.
        self.graph
            .edges(node)
            .filter(|edge| edge.target() != node)
            .for_each(|edge| {
                let community = self.memberships[edge.target().index()].unwrap();
                *communities.entry(community).or_default() += *edge.weight();
            });

        communities
    }

    /// Returns the best community to put a node in when there is one.
    ///
    /// When no better options than the current community is found, `None` is returned.
    fn best_community(&self, node: Ix) -> Option<usize> {
        // Find the best community and its modularity change.
        let (best, delta) = self
            // Consider each neighboring community.
            .incident_communities(node)
            .iter()
            // Calculate the modularity change for moving into the community.
            .map(|(&community, &k_i_in)| (community, self.delta(node, community, k_i_in)))
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
    fn remove(&mut self, node: Ix) {
        // Get the community the node was in.
        let community = self.memberships[node.index()]
            .expect("Node to be removed should be part of a community.");

        // Remove the community membership.
        self.memberships[node.index()] = None;

        // Decrease the total weight of the community by the node weight.
        self.sigma_tot[community] -= self.node_weights[node.index()];
    }

    /// Inserts a node into the given community.
    fn insert(&mut self, node: Ix, community: usize) {
        // Enter the community membership.
        self.memberships[node.index()] = Some(community);

        // Increase the total weight of the community by the node weight.
        self.sigma_tot[community] += self.node_weights[node.index()];
    }

    /// Moves a node into the best community, returning whether a change actually occurred.
    ///
    /// Removes the node, finds the best community and places the node there.
    fn move_node(&mut self, node: Ix) -> bool {
        // Note the current community and then remove the node.
        let current = self.memberships[node.index()]
            .expect("Node to be moved should be part of a community.");

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
    use petgraph::csr::Csr;
    use petgraph::Undirected;

    /// Generates the example graph from the [Louvain paper][paper].
    ///
    /// [paper]: https://perso.uclouvain.be/vincent.blondel/publications/08BG.pdf
    fn graph() -> Csr<(), u8, Undirected, u8> {
        let edges = [
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
        ];

        let mut csr_edges: Vec<(u8, u8, u8)> =
            edges.iter().copied().map(|(a, b)| (a, b, 1)).collect();

        csr_edges.extend(edges.into_iter().map(|(a, b)| (b, a, 1)));
        csr_edges.sort_unstable();
        Csr::from_sorted_edges(&csr_edges).unwrap()
    }

    #[test]
    fn clustering() {
        assert_eq!(
            louvain(&graph()),
            vec![
                vec![
                    vec![0],
                    vec![1],
                    vec![2],
                    vec![3],
                    vec![4],
                    vec![5],
                    vec![6],
                    vec![7],
                    vec![8],
                    vec![9],
                    vec![10],
                    vec![11],
                    vec![12],
                    vec![13],
                    vec![14],
                    vec![15],
                ],
                vec![
                    vec![3, 6, 7],
                    vec![0, 1, 2, 4, 5,],
                    vec![8, 9, 10, 12, 14, 15,],
                    vec![11, 13],
                ],
                vec![
                    vec![3, 6, 7, 0, 1, 2, 4, 5,],
                    vec![8, 9, 10, 12, 14, 15, 11, 13],
                ],
            ]
        );
    }
}
