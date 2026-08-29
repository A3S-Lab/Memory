use super::{MemoryNode, MemoryRelation, MemoryRelationKind, MemoryRepositoryError, MemoryStatus};
use std::collections::BTreeMap;

pub(crate) fn validate_graph_node(
    node: &MemoryNode,
    staged: &BTreeMap<String, MemoryNode>,
    base: Option<&BTreeMap<String, MemoryNode>>,
) -> Result<(), MemoryRepositoryError> {
    if node.status == MemoryStatus::Superseded
        && !node
            .relations
            .iter()
            .any(|relation| relation.kind == MemoryRelationKind::SupersededBy)
    {
        return Err(MemoryRepositoryError::invariant(format!(
            "superseded memory node {} requires a superseded_by relation",
            node.id
        )));
    }
    if node.status == MemoryStatus::Conflicted
        && !node
            .relations
            .iter()
            .any(|relation| relation.kind == MemoryRelationKind::ConflictsWith)
    {
        return Err(MemoryRepositoryError::invariant(format!(
            "conflicted memory node {} requires a conflicts_with relation",
            node.id
        )));
    }

    for relation in &node.relations {
        let Some(target) = find_node(staged, base, &relation.target_id) else {
            return Err(MemoryRepositoryError::RelationTargetNotFound {
                node_id: node.id.clone(),
                target_id: relation.target_id.clone(),
            });
        };
        let inverse = MemoryRelation::new(relation.kind.inverse(), &node.id);
        if !target.relations.contains(&inverse) {
            return Err(MemoryRepositoryError::invariant(format!(
                "relation {:?} from {} to {} is missing its {:?} counterpart",
                relation.kind, node.id, target.id, inverse.kind
            )));
        }
        if relation.kind == MemoryRelationKind::Supersedes
            && target.status != MemoryStatus::Superseded
        {
            return Err(MemoryRepositoryError::invariant(format!(
                "memory node {} supersedes {}, but the target is not superseded",
                node.id, target.id
            )));
        }
    }
    Ok(())
}

pub(crate) fn find_node<'a>(
    staged: &'a BTreeMap<String, MemoryNode>,
    base: Option<&'a BTreeMap<String, MemoryNode>>,
    node_id: &str,
) -> Option<&'a MemoryNode> {
    staged
        .get(node_id)
        .or_else(|| base.and_then(|nodes| nodes.get(node_id)))
}
