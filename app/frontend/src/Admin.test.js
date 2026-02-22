import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import Admin from './Admin';

const mockUsers = { users: [{ id: 1, username: 'admin', role: 'admin' }, { id: 2, username: 'user', role: 'read_only' }] };

const mockGet = jest.fn();

jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => jest.fn(),
}));

jest.mock('./tokenManager', () => ({
  __esModule: true,
  default: {
    get: (...args) => mockGet(...args),
    clearToken: jest.fn(),
  },
}));

describe('Admin page', () => {
  beforeEach(() => {
    mockGet.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockUsers),
    });
    Object.defineProperty(window, 'localStorage', {
      value: {
        getItem: jest.fn((key) => (key === 'user' ? JSON.stringify({ username: 'admin', role: 'admin' }) : null)),
        setItem: jest.fn(),
        removeItem: jest.fn(),
        clear: jest.fn(),
      },
      writable: true,
    });
  });

  it('renders Admin with User Management and Past Conversations tabs', async () => {
    render(
      <MemoryRouter>
        <Admin />
      </MemoryRouter>
    );
    await waitFor(() => {
      expect(screen.getByText('User Management')).toBeInTheDocument();
    });
    expect(screen.getByText('Past Conversations')).toBeInTheDocument();
  });

  it('shows User Management content by default', async () => {
    render(
      <MemoryRouter>
        <Admin />
      </MemoryRouter>
    );
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /add new user/i })).toBeInTheDocument();
    });
  });

  it('shows Past Conversations tab content when Past Conversations tab is clicked', async () => {
    render(
      <MemoryRouter>
        <Admin />
      </MemoryRouter>
    );
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /past conversations/i })).toBeInTheDocument();
    });
    await userEvent.click(screen.getByRole('button', { name: /past conversations/i }));
    expect(screen.getByText(/select one or more users to view their conversation history/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /load conversations/i })).toBeInTheDocument();
  });

  it('Past Conversations tab shows user checkboxes and Load conversations button', async () => {
    render(
      <MemoryRouter>
        <Admin />
      </MemoryRouter>
    );
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /past conversations/i })).toBeInTheDocument();
    });
    await userEvent.click(screen.getByRole('button', { name: /past conversations/i }));
    expect(screen.getByRole('button', { name: /load conversations/i })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'admin' })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'user' })).toBeInTheDocument();
  });
});
