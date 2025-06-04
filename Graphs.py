import matplotlib.pyplot as plt

# Create a scatter plot
def scatter_plot(X_obs, Y_obs, X_domain=None,  Y_hat=None, X_unk=None, Y_unk=None, ci_lower_pred=None, ci_upper_pred=None, filename=None):
    plt.scatter(X_obs, Y_obs, color="blue")
    if Y_hat is not None:
        plt.scatter(X_domain, Y_hat,  color='red')
    if ci_lower_pred is not None:
        plt.scatter(X_domain, ci_lower_pred,  color='orange')
        plt.scatter(X_domain, ci_upper_pred,  color='orange')
    if Y_unk is not None:
        plt.scatter(X_unk, Y_unk, color="green", marker='^')
    # Add labels and title
    plt.xlabel('yearly income')
    plt.ylabel('yearly spending (EUR)')
    plt.title('Luxury spending')
    # save the plot
    plt.savefig(filename, bbox_inches='tight')
    # Display the plot
    plt.show()