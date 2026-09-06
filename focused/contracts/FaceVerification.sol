// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title FaceVerification
/// @notice Anchors SHA-256 fingerprints of externally-discovered content on-chain.
/// @dev Only the cryptographic fingerprint plus small metadata is stored.
///      Images are never stored on-chain.
contract FaceVerification {
    struct VerificationRecord {
        bytes32 dataHash;    // sha256 of the canonical post metadata JSON
        bytes32 imageHash;   // sha256 of the matched image bytes (0x0 if unavailable)
        string sourceUrl;    // where the match was found
        uint256 timestamp;   // block timestamp of storage
        address submitter;   // msg.sender
    }

    struct BatchAnchor {
        bytes32 root;          // merkle root over the batch's dataHashes
        uint256 firstRecordId; // index of the first record in the batch
        uint256 count;         // number of records in the batch
        uint256 timestamp;
        address submitter;
    }

    VerificationRecord[] private _records;
    BatchAnchor[] private _batches;

    /// @notice index+1 of the most recent record for a given dataHash (0 = none)
    mapping(bytes32 => uint256) private _latestByHash;

    event RecordStored(
        bytes32 indexed dataHash,
        bytes32 imageHash,
        string sourceUrl,
        uint256 timestamp,
        address indexed submitter,
        uint256 recordId
    );

    event BatchStored(
        bytes32 indexed batchRoot,
        uint256 indexed batchId,
        uint256 firstRecordId,
        uint256 count,
        address indexed submitter,
        uint256 timestamp
    );

    error EmptyHash();
    error EmptySourceUrl();
    error RecordNotFound();
    error BatchNotFound();
    error LengthMismatch();
    error EmptyBatch();
    error BatchTooLarge();

    /// @notice Store a verification record.
    /// @return recordId index of the stored record
    function storeRecord(
        bytes32 dataHash,
        bytes32 imageHash,
        string calldata sourceUrl
    ) external returns (uint256 recordId) {
        if (dataHash == bytes32(0)) revert EmptyHash();
        if (bytes(sourceUrl).length == 0) revert EmptySourceUrl();

        _records.push(
            VerificationRecord({
                dataHash: dataHash,
                imageHash: imageHash,
                sourceUrl: sourceUrl,
                timestamp: block.timestamp,
                submitter: msg.sender
            })
        );

        recordId = _records.length - 1;
        _latestByHash[dataHash] = recordId + 1;

        emit RecordStored(dataHash, imageHash, sourceUrl, block.timestamp, msg.sender, recordId);
    }

    /// @notice Store many verification records in a single transaction.
    /// @dev Amortises the ~21k base gas cost across the whole batch; the merkle
    ///      root lets anyone prove membership of an individual record later.
    /// @param batchRoot merkle root computed off-chain over dataHashes (may be 0)
    /// @return firstRecordId id of the first record written
    /// @return batchId id of the stored batch anchor
    function storeBatch(
        bytes32 batchRoot,
        bytes32[] calldata dataHashes,
        bytes32[] calldata imageHashes,
        string[] calldata sourceUrls
    ) external returns (uint256 firstRecordId, uint256 batchId) {
        uint256 n = dataHashes.length;
        if (n == 0) revert EmptyBatch();
        if (n > 200) revert BatchTooLarge();
        if (imageHashes.length != n || sourceUrls.length != n) revert LengthMismatch();

        firstRecordId = _records.length;
        for (uint256 i = 0; i < n; ) {
            bytes32 dh = dataHashes[i];
            if (dh == bytes32(0)) revert EmptyHash();
            if (bytes(sourceUrls[i]).length == 0) revert EmptySourceUrl();

            _records.push(
                VerificationRecord({
                    dataHash: dh,
                    imageHash: imageHashes[i],
                    sourceUrl: sourceUrls[i],
                    timestamp: block.timestamp,
                    submitter: msg.sender
                })
            );
            _latestByHash[dh] = firstRecordId + i + 1;
            emit RecordStored(dh, imageHashes[i], sourceUrls[i], block.timestamp, msg.sender, firstRecordId + i);
            unchecked { ++i; }
        }

        _batches.push(
            BatchAnchor({
                root: batchRoot,
                firstRecordId: firstRecordId,
                count: n,
                timestamp: block.timestamp,
                submitter: msg.sender
            })
        );
        batchId = _batches.length - 1;
        emit BatchStored(batchRoot, batchId, firstRecordId, n, msg.sender, block.timestamp);
    }

    /// @notice Read a batch anchor back by id.
    function getBatch(uint256 batchId)
        external
        view
        returns (bytes32 root, uint256 firstRecordId, uint256 count, uint256 timestamp, address submitter)
    {
        if (batchId >= _batches.length) revert BatchNotFound();
        BatchAnchor storage b = _batches[batchId];
        return (b.root, b.firstRecordId, b.count, b.timestamp, b.submitter);
    }

    /// @notice Bulk read-back: all dataHashes in [start, start+count).
    /// @dev One RPC call verifies an entire batch instead of N calls.
    function getDataHashes(uint256 start, uint256 count) external view returns (bytes32[] memory hashes) {
        if (start + count > _records.length) revert RecordNotFound();
        hashes = new bytes32[](count);
        for (uint256 i = 0; i < count; ) {
            hashes[i] = _records[start + i].dataHash;
            unchecked { ++i; }
        }
    }

    function totalBatches() external view returns (uint256) {
        return _batches.length;
    }

    /// @notice Read a record back by id.
    function getRecord(uint256 recordId)
        external
        view
        returns (bytes32 dataHash, bytes32 imageHash, string memory sourceUrl, uint256 timestamp, address submitter)
    {
        if (recordId >= _records.length) revert RecordNotFound();
        VerificationRecord storage r = _records[recordId];
        return (r.dataHash, r.imageHash, r.sourceUrl, r.timestamp, r.submitter);
    }

    /// @notice Read the latest record stored for a specific fingerprint.
    function getLatestByHash(bytes32 dataHash)
        external
        view
        returns (bytes32 imageHash, string memory sourceUrl, uint256 timestamp, address submitter, uint256 recordId)
    {
        uint256 slot = _latestByHash[dataHash];
        if (slot == 0) revert RecordNotFound();
        recordId = slot - 1;
        VerificationRecord storage r = _records[recordId];
        return (r.imageHash, r.sourceUrl, r.timestamp, r.submitter, recordId);
    }

    /// @notice True if the fingerprint has ever been anchored.
    function exists(bytes32 dataHash) external view returns (bool) {
        return _latestByHash[dataHash] != 0;
    }

    function totalRecords() external view returns (uint256) {
        return _records.length;
    }
}
