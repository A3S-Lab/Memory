use super::graph::{find_node, validate_graph_node};
use super::validation::{
    validate_count, validate_required_text, validate_unit_float, MAX_CONTENT_BYTES,
    MAX_EVIDENCE_PER_NODE, MAX_IDENTIFIER_BYTES, MAX_RELATIONS_PER_NODE,
};
use super::{
    EvidenceRef, MemoryChangeSet, MemoryNode, MemoryOperation, MemoryRepositoryError,
    MemoryRevisionKind, MemoryStatus, RevisionMode,
};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) struct StagedChange {
    pub(crate) nodes: BTreeMap<String, MemoryNode>,
    pub(crate) changed: BTreeSet<String>,
}

pub(crate) fn stage_change_set(
    change_set: &MemoryChangeSet,
    base: Option<&BTreeMap<String, MemoryNode>>,
) -> Result<StagedChange, MemoryRepositoryError> {
    let mut nodes = BTreeMap::new();
    let mut changed = BTreeSet::new();
    let mut relation_neighbors = BTreeSet::new();
    for operation in &change_set.operations {
        apply_operation(
            operation,
            change_set,
            base,
            &mut nodes,
            &mut changed,
            &mut relation_neighbors,
        )?;
    }

    for node_id in &changed {
        let node = nodes.get(node_id).ok_or_else(|| {
            MemoryRepositoryError::invariant(format!(
                "changed memory node {node_id} was not staged"
            ))
        })?;
        node.validate()?;
        validate_graph_node(node, &nodes, base)?;
    }
    for node_id in relation_neighbors.difference(&changed) {
        if let Some(node) = find_node(&nodes, base, node_id) {
            node.validate()?;
            validate_graph_node(node, &nodes, base)?;
        }
    }

    Ok(StagedChange { nodes, changed })
}

fn apply_operation(
    operation: &MemoryOperation,
    change_set: &MemoryChangeSet,
    base: Option<&BTreeMap<String, MemoryNode>>,
    staged: &mut BTreeMap<String, MemoryNode>,
    changed: &mut BTreeSet<String>,
    relation_neighbors: &mut BTreeSet<String>,
) -> Result<(), MemoryRepositoryError> {
    match operation {
        MemoryOperation::Create { node: draft } => {
            if draft.namespace != change_set.namespace {
                return Err(MemoryRepositoryError::NamespaceMismatch {
                    context: format!("create operation for node {}", draft.id),
                });
            }
            if !matches!(draft.status, MemoryStatus::Candidate | MemoryStatus::Active) {
                return Err(MemoryRepositoryError::InvalidTransition {
                    node_id: draft.id.clone(),
                    from: "absent".into(),
                    to: draft.status.as_str().into(),
                });
            }
            if draft.created_at > change_set.occurred_at {
                return Err(MemoryRepositoryError::invalid(
                    "node.createdAt",
                    "must not follow changeSet.occurredAt",
                ));
            }
            if staged.contains_key(&draft.id)
                || base.is_some_and(|nodes| nodes.contains_key(&draft.id))
            {
                return Err(MemoryRepositoryError::NodeAlreadyExists {
                    node_id: draft.id.clone(),
                });
            }
            validate_evidence_times(&draft.evidence, change_set)?;
            let node = MemoryNode::from_draft(draft.clone());
            node.validate()?;
            relation_neighbors.extend(
                node.relations
                    .iter()
                    .map(|relation| relation.target_id.clone()),
            );
            changed.insert(node.id.clone());
            staged.insert(node.id.clone(), node);
        }
        MemoryOperation::Activate {
            node_id,
            expected_revision,
        } => {
            let node = node_for_update(node_id, *expected_revision, base, staged)?;
            if node.status != MemoryStatus::Candidate {
                return Err(invalid_transition(node, MemoryStatus::Active));
            }
            if node.evidence.is_empty() {
                return Err(MemoryRepositoryError::EvidenceRequired {
                    node_id: node.id.clone(),
                });
            }
            node.begin_revision(change_set.occurred_at, MemoryRevisionKind::Activated)?;
            node.status = MemoryStatus::Active;
            changed.insert(node_id.clone());
        }
        MemoryOperation::Corroborate {
            node_id,
            expected_revision,
            evidence,
        } => {
            if evidence.is_empty() {
                return Err(MemoryRepositoryError::EvidenceRequired {
                    node_id: node_id.clone(),
                });
            }
            validate_evidence_times(evidence, change_set)?;
            let node = node_for_update(node_id, *expected_revision, base, staged)?;
            ensure_evolvable(node, "corroborate")?;
            validate_new_evidence(node, evidence)?;
            node.begin_revision(change_set.occurred_at, MemoryRevisionKind::Corroborated)?;
            node.evidence.extend(evidence.iter().cloned());
            node.evidence.sort();
            changed.insert(node_id.clone());
        }
        MemoryOperation::Revise {
            node_id,
            expected_revision,
            content,
            mode,
            evidence,
            confidence,
            importance,
        } => {
            validate_required_text("operation.content", content, MAX_CONTENT_BYTES)?;
            if evidence.is_empty() {
                return Err(MemoryRepositoryError::EvidenceRequired {
                    node_id: node_id.clone(),
                });
            }
            validate_evidence_times(evidence, change_set)?;
            if let Some(value) = confidence {
                validate_unit_float("operation.confidence", *value)?;
            }
            if let Some(value) = importance {
                validate_unit_float("operation.importance", *value)?;
            }
            let node = node_for_update(node_id, *expected_revision, base, staged)?;
            ensure_evolvable(node, "revise")?;
            validate_new_evidence(node, evidence)?;
            let revision_kind = match mode {
                RevisionMode::Refinement => MemoryRevisionKind::Refined,
                RevisionMode::Correction => MemoryRevisionKind::Corrected,
            };
            node.begin_revision(change_set.occurred_at, revision_kind)?;
            node.content.clone_from(content);
            node.evidence.extend(evidence.iter().cloned());
            node.evidence.sort();
            if let Some(value) = confidence {
                node.confidence = *value;
            }
            if let Some(value) = importance {
                node.importance = *value;
            }
            changed.insert(node_id.clone());
        }
        MemoryOperation::AddRelation {
            node_id,
            expected_revision,
            relation,
        } => {
            relation.validate(node_id)?;
            let node = node_for_update(node_id, *expected_revision, base, staged)?;
            ensure_evolvable(node, "add a relation to")?;
            if node.relations.contains(relation) {
                return Err(MemoryRepositoryError::invariant(format!(
                    "memory node {node_id} already contains the relation"
                )));
            }
            validate_count(
                "node relations",
                node.relations.len() + 1,
                MAX_RELATIONS_PER_NODE,
            )?;
            node.begin_revision(change_set.occurred_at, MemoryRevisionKind::RelationChanged)?;
            node.relations.push(relation.clone());
            node.relations.sort();
            relation_neighbors.insert(relation.target_id.clone());
            changed.insert(node_id.clone());
        }
        MemoryOperation::RemoveRelation {
            node_id,
            expected_revision,
            relation,
        } => {
            relation.validate(node_id)?;
            let node = node_for_update(node_id, *expected_revision, base, staged)?;
            ensure_evolvable(node, "remove a relation from")?;
            let Some(index) = node.relations.iter().position(|item| item == relation) else {
                return Err(MemoryRepositoryError::invariant(format!(
                    "memory node {node_id} does not contain the relation"
                )));
            };
            node.begin_revision(change_set.occurred_at, MemoryRevisionKind::RelationChanged)?;
            node.relations.remove(index);
            relation_neighbors.insert(relation.target_id.clone());
            changed.insert(node_id.clone());
        }
        MemoryOperation::SetStatus {
            node_id,
            expected_revision,
            status,
        } => {
            if !matches!(
                status,
                MemoryStatus::Superseded | MemoryStatus::Conflicted | MemoryStatus::Tombstoned
            ) {
                return Err(MemoryRepositoryError::invalid(
                    "operation.status",
                    "use Activate for candidate-to-active transitions",
                ));
            }
            let node = node_for_update(node_id, *expected_revision, base, staged)?;
            if !valid_status_transition(node.status, *status) {
                return Err(invalid_transition(node, *status));
            }
            node.begin_revision(change_set.occurred_at, MemoryRevisionKind::StatusChanged)?;
            node.status = *status;
            changed.insert(node_id.clone());
        }
    }
    Ok(())
}

fn node_for_update<'a>(
    node_id: &str,
    expected_revision: u64,
    base: Option<&BTreeMap<String, MemoryNode>>,
    staged: &'a mut BTreeMap<String, MemoryNode>,
) -> Result<&'a mut MemoryNode, MemoryRepositoryError> {
    validate_required_text("operation.nodeId", node_id, MAX_IDENTIFIER_BYTES)?;
    if !staged.contains_key(node_id) {
        let node = base
            .and_then(|nodes| nodes.get(node_id))
            .cloned()
            .ok_or_else(|| MemoryRepositoryError::NodeNotFound {
                node_id: node_id.to_owned(),
            })?;
        staged.insert(node_id.to_owned(), node);
    }
    let node = staged.get_mut(node_id).ok_or_else(|| {
        MemoryRepositoryError::invariant(format!("memory node {node_id} was not staged for update"))
    })?;
    if node.revision != expected_revision {
        return Err(MemoryRepositoryError::RevisionConflict {
            node_id: node_id.to_owned(),
            expected: expected_revision,
            actual: node.revision,
        });
    }
    Ok(node)
}

fn validate_evidence_times(
    evidence: &[EvidenceRef],
    change_set: &MemoryChangeSet,
) -> Result<(), MemoryRepositoryError> {
    for item in evidence {
        item.validate()?;
        if item.occurred_at > change_set.occurred_at {
            return Err(MemoryRepositoryError::invalid(
                "evidence.occurredAt",
                "must not follow changeSet.occurredAt",
            ));
        }
    }
    Ok(())
}

fn validate_new_evidence(
    node: &MemoryNode,
    evidence: &[EvidenceRef],
) -> Result<(), MemoryRepositoryError> {
    validate_count(
        "node evidence",
        node.evidence.len() + evidence.len(),
        MAX_EVIDENCE_PER_NODE,
    )?;
    let additions = evidence
        .iter()
        .map(|item| item.uri.as_str())
        .collect::<BTreeSet<_>>();
    if additions.len() != evidence.len()
        || evidence
            .iter()
            .any(|item| node.evidence.iter().any(|known| known.uri == item.uri))
    {
        return Err(MemoryRepositoryError::invariant(format!(
            "memory node {} cannot add duplicate evidence URIs",
            node.id
        )));
    }
    Ok(())
}

fn ensure_evolvable(node: &MemoryNode, operation: &str) -> Result<(), MemoryRepositoryError> {
    if matches!(
        node.status,
        MemoryStatus::Superseded | MemoryStatus::Tombstoned
    ) {
        return Err(MemoryRepositoryError::invariant(format!(
            "cannot {operation} frozen memory node {} in {} status",
            node.id,
            node.status.as_str()
        )));
    }
    Ok(())
}

fn valid_status_transition(from: MemoryStatus, to: MemoryStatus) -> bool {
    matches!(
        (from, to),
        (MemoryStatus::Candidate, MemoryStatus::Conflicted)
            | (MemoryStatus::Candidate, MemoryStatus::Tombstoned)
            | (MemoryStatus::Active, MemoryStatus::Superseded)
            | (MemoryStatus::Active, MemoryStatus::Conflicted)
            | (MemoryStatus::Active, MemoryStatus::Tombstoned)
            | (MemoryStatus::Conflicted, MemoryStatus::Superseded)
            | (MemoryStatus::Conflicted, MemoryStatus::Tombstoned)
            | (MemoryStatus::Superseded, MemoryStatus::Tombstoned)
    )
}

fn invalid_transition(node: &MemoryNode, to: MemoryStatus) -> MemoryRepositoryError {
    MemoryRepositoryError::InvalidTransition {
        node_id: node.id.clone(),
        from: node.status.as_str().into(),
        to: to.as_str().into(),
    }
}
