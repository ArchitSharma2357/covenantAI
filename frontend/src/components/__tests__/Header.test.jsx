import React from 'react'
import { render, screen } from '@testing-library/react'
import Header from '../Header'

describe('Header', () => {
  it('renders search input and upload button', () => {
    render(<Header isAuthenticated={false} user={null} onLogout={() => {}} />)
    const input = screen.getByPlaceholderText(/Search documents/i)
    expect(input).toBeInTheDocument()
    const uploadBtn = screen.getByTitle(/Upload document/i)
    expect(uploadBtn).toBeInTheDocument()
  })
})
