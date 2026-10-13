export function serializeSearchRequest(personId, request) {
  const value = personId.trim()
  if (!/^-?[0-9]+$/.test(value)) {
    throw new Error('Enter an integer patient ID.')
  }

  // JSON.stringify would round IDs beyond JavaScript's safe integer range.
  const id = BigInt(value).toString()
  return `{"person_id":${id},"request":${JSON.stringify(request)}}`
}
