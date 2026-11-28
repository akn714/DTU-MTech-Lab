# ASNS End-Semester: Final Topic Checklist

## Unit 1 – Security Fundamentals

- [ ] CIA Triad – Confidentiality, Integrity, Availability.
- [ ] Security goals and principles.
- [ ] Security services and security mechanisms.
- [ ] Security attacks – Passive and Active.
- [ ] Communication security and secure system design.
- [ ] Cryptography and Steganography.
- [ ] Cryptanalysis.
- [ ] Symmetric vs Asymmetric encryption.
- [ ] Public and Private keys.
- [ ] Intruders, viruses, worms and related threats.

**Reference:**
- Stallings – Chapter 1 (Full).
- Stallings – Chapter 2 (Section 2.1).

## Unit 2 – Classical Encryption Techniques

- [ ] Substitution ciphers.
  - [ ] Caesar Cipher.
  - [ ] Modified Caesar Cipher.
  - [ ] Monoalphabetic Cipher.
  - [ ] Playfair Cipher.
  - [ ] Vigenère Cipher.
  - [ ] Vernam Cipher.
  - [ ] One-Time Pad.
  - [ ] Hill Cipher.
- [ ] Transposition ciphers.
- [ ] Cryptanalysis of classical ciphers.
- [ ] Stream ciphers.
- [ ] Block ciphers.
- [ ] Pseudorandom numbers – Overview.

**Reference:**
- Stallings – Chapter 2 (Full).
- Stallings – Chapter 7 (Stream cipher and pseudorandom number overview).

## Unit 3 – DES and Block Cipher Algorithms

- [ ] Modern block cipher principles.
- [ ] Confusion and Diffusion (Shannon's theory).
- [ ] Feistel structure.
- [ ] DES algorithm.
  - [ ] Initial and Final Permutation.
  - [ ] 16 rounds.
  - [ ] Expansion permutation.
  - [ ] XOR operation.
  - [ ] S-Boxes.
  - [ ] Key generation.
- [ ] Strength and weaknesses of DES.
- [ ] Double DES.
- [ ] Triple DES.
- [ ] Block cipher modes of operation:
  - [ ] ECB.
  - [ ] CBC.
  - [ ] CFB.
  - [ ] OFB.
  - [ ] CTR.
- [ ] AES algorithm.

**Reference:**
- Stallings – Chapter 3 (Full).
- Stallings – Chapter 6 (Full).
- Stallings – Chapter 5 (AES; additional chapter).

## Unit 4 – IDEA Algorithm

- [ ] Introduction and working of IDEA.
- [ ] Block size and key size.
- [ ] Key generation.
- [ ] Eight rounds.
- [ ] Subkey distribution.
- [ ] Mathematical operations.
- [ ] Encryption and decryption.
- [ ] Strength of IDEA.

**Reference:**
- Separate preparation required. Do not assume that the selected Stallings chapters cover IDEA.

## Unit 5 – Number Theory and RSA

- [ ] Euclidean algorithm.
- [ ] GCD and coprime numbers.
- [ ] Modular arithmetic.
- [ ] Prime numbers.
- [ ] Euler's Totient Function.
- [ ] Modular multiplicative inverse.
- [ ] Public-key cryptography principles.
- [ ] RSA algorithm.
  - [ ] Key generation.
  - [ ] Public and private keys.
  - [ ] Encryption.
  - [ ] Decryption.
  - [ ] Numerical problems.

**Reference:**
- Stallings – Chapter 4 (Euclidean algorithm).
- Stallings – Chapter 8 (Essential number theory sections).
- Stallings – Chapter 9 (Full).

## Unit 6 – Diffie-Hellman and Digital Signatures

- [ ] Diffie-Hellman Key Exchange.
- [ ] Primitive roots.
- [ ] Public values.
- [ ] Shared secret key generation.
- [ ] Man-in-the-Middle attack.
- [ ] Digital signatures.
- [ ] Authentication.
- [ ] Integrity.
- [ ] Non-repudiation.
- [ ] RSA Digital Signature.
- [ ] DSS (Digital Signature Standard).

**Reference:**
- Stallings – Chapter 10 (Full).
- Stallings – Chapter 13 (Except 13.3).

## Unit 7 – Hash Functions and Message Authentication

- [ ] Hash functions and their properties.
- [ ] Message digest.
- [ ] Collision resistance.
- [ ] Preimage and second-preimage resistance.
- [ ] SHA algorithm.
- [ ] SHA-512:
  - [ ] Padding.
  - [ ] Message schedule.
  - [ ] Initial hash values.
  - [ ] 80 rounds.
  - [ ] Compression function.
- [ ] Message Authentication Code (MAC).
- [ ] HMAC.
  - [ ] HMAC structure.
  - [ ] ipad and opad.
  - [ ] Key processing.
  - [ ] Working of HMAC.

**Reference:**
- Stallings – Chapter 11 (Full).
- Stallings – Chapter 12 (Full).

## Unit 8 – Key Management and Authentication

- [ ] Symmetric key distribution.
- [ ] Public key distribution.
- [ ] Public-Key Authority.
- [ ] Digital Certificates.
- [ ] X.509 Certificates.
- [ ] Certificate Authority (CA).
- [ ] Certificate Revocation.
- [ ] User authentication.
- [ ] Kerberos:
  - [ ] Authentication Server (AS).
  - [ ] Ticket Granting Server (TGS).
  - [ ] Ticket Granting Ticket (TGT).
  - [ ] Service Ticket.
  - [ ] Kerberos message exchange.

**Reference:**
- Stallings – Chapter 14 (Sections 14.1–14.4).
- Stallings – Chapter 15 (Only 15.3).

## Unit 9 – IP Security (IPSec)

- [ ] Introduction to IP Security.
- [ ] IPSec architecture.
- [ ] Transport Mode.
- [ ] Tunnel Mode.
- [ ] Authentication Header (AH).
- [ ] Encapsulating Security Payload (ESP).
- [ ] Security Association (SA).
- [ ] Security Association Database (SAD).
- [ ] Security Policy Database (SPD).
- [ ] Replay protection.
- [ ] Combining Security Associations.
- [ ] Internet Key Exchange (IKE).
- [ ] IKE phases.
- [ ] ISAKMP.
- [ ] Oakley and SKEME overview.
- [ ] Nonces and Cookies.
- [ ] Perfect Forward Secrecy (PFS).
- [ ] AH vs ESP comparison.

**Reference:**
- Forouzan – Chapter 18 (Full).
- Stallings – Chapter 19 (Additional reference).

## Unit 10 – SSL, TLS and Web Security

- [ ] Web security requirements.
- [ ] SSL architecture.
- [ ] SSL Record Protocol.
- [ ] SSL Handshake Protocol.
- [ ] Change Cipher Spec Protocol.
- [ ] Alert Protocol.
- [ ] SSL session and connection.
- [ ] TLS architecture and working.
- [ ] SSL vs TLS.
- [ ] Cipher suites.
- [ ] Client Hello and Server Hello.
- [ ] Key exchange.
- [ ] Server and client authentication.
- [ ] Handshake completion.
- [ ] SSL/TLS handshake diagram.

**Reference:**
- Forouzan – Chapter 17:
  - [ ] 17.1–17.3: Overview/revision.
  - [ ] 17.4: Deep study.
- Stallings – Chapter 16 (Additional reference).

## Unit 11 – Additional Network Security Topics

- [ ] VPN – Architecture, working and diagram.
- [ ] Firewall design principles.
- [ ] Types of Firewalls:
  - [ ] Packet Filtering.
  - [ ] Stateful Inspection.
  - [ ] Application-Level Gateway.
  - [ ] Circuit-Level Gateway.
- [ ] Intrusion Detection System (IDS).
- [ ] Trusted Systems.
- [ ] SET (Secure Electronic Transaction).
- [ ] Electronic Money.
- [ ] WAP Security.
- [ ] Malware, viruses and worms.
- [ ] TSP (Time-Stamp Protocol).
- [ ] SSL vs TSP comparison.

**Reference:**
- Stallings – Chapters 17, 20, 21 and 22 (selected topics).
- Class notes and additional course material for SET, electronic money, VPN and TSP.

---

# 2. Book Reference Mapping

## William Stallings

- [ ] Chapter 1 – Full.
- [ ] Chapter 2 – Full.
- [ ] Chapter 3 – Full.
- [ ] Chapter 4 – Euclidean Algorithm.
- [ ] Chapter 5 – AES (Additional).
- [ ] Chapter 6 – Full.
- [ ] Chapter 7 – Stream cipher and pseudorandom numbers overview.
- [ ] Chapter 8 – Essential number theory sections (Additional).
- [ ] Chapter 9 – Full.
- [ ] Chapter 10 – Full.
- [ ] Chapter 11 – Full.
- [ ] Chapter 12 – Full.
- [ ] Chapter 13 – Except 13.3.
- [ ] Chapter 14 – Sections 14.1–14.4.
- [ ] Chapter 15 – Only 15.3.

## Forouzan

- [ ] Chapter 17 – Security at the Transport Layer: SSL and TLS.
  - [ ] Sections 17.1–17.3: Read overview.
  - [ ] Section 17.4: Study deeply.
- [ ] Chapter 18 – IP Security (Full).

---

# 3. Important Topics Missing From Original Book Selection

These topics need separate preparation or additional chapters.

- [ ] IDEA – Algorithm, encryption, decryption and key generation.
- [ ] AES – Stallings Chapter 5.
- [ ] SHA-512 – Specific algorithm and its 80 rounds.
- [ ] Number Theory – Essential sections of Stallings Chapter 8.
- [ ] TSP – Time-Stamp Protocol.
- [ ] SET – Secure Electronic Transaction.
- [ ] Electronic Money.
- [ ] Firewall Types and Design Principles.
- [ ] IDS – Intrusion Detection System.
- [ ] Trusted Systems.
- [ ] Malware and Intruders.
- [ ] VPN – Architecture and working.
- [ ] WAP Security.

---

# 4. Final Preparation Priority

## Priority 1 – Must Prepare Thoroughly

- [ ] RSA.
- [ ] DES.
- [ ] Double DES.
- [ ] Triple DES.
- [ ] Block Cipher Modes.
- [ ] IDEA.
- [ ] AES.
- [ ] SHA-512.
- [ ] HMAC.
- [ ] Diffie-Hellman.
- [ ] Digital Signatures.
- [ ] IPSec.
- [ ] AH.
- [ ] ESP.
- [ ] SSL/TLS Handshake.

## Priority 2 – Important

- [ ] Classical Ciphers.
- [ ] Euclidean Algorithm.
- [ ] Number Theory.
- [ ] X.509 Certificates.
- [ ] Kerberos.
- [ ] Key Distribution.
- [ ] IKE.
- [ ] VPN.
- [ ] Firewall Types.

## Priority 3 – Remaining Syllabus

- [ ] CIA Triad.
- [ ] Security Services.
- [ ] Security Attacks.
- [ ] Steganography.
- [ ] Malware.
- [ ] IDS.
- [ ] Trusted Systems.
- [ ] SET.
- [ ] Electronic Money.
- [ ] WAP Security.
- [ ] TSP.

---

# 5. Important Preparation Notes

1. Do not skip Stallings Chapter 8 entirely. Its number theory is important for understanding RSA and Diffie-Hellman.
2. Do not assume the selected chapters cover the entire syllabus. IDEA, AES and additional network security topics need separate preparation.
3. For algorithms, prepare their working diagrams, steps, key sizes, block sizes and numerical examples wherever applicable.
4. Follow the 11 units in the given sequence to maintain conceptual continuity.
